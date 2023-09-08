import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import re
from sentence_transformers import SentenceTransformer, util

torch.set_num_threads(8)
torch.inference_mode()
torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
torch.backends.cuda.sdp_kernel(enable_flash=True,
                               enable_math=False,
                               enable_mem_efficient=True)

ABCDE = 'ABCDE'
PROMPT = 'prompt'
ANSWER = 'answer'

TIER_1_QUESTION_EMBEDDINGS = "embeddings"
TIER_1_QUERIES = 'tier_1_queries'
TIER_1_HITS = 'tier_1_hits'
TIER_2_HITS = 'tier_2_hits'

TIER_1_PASSAGES = 'tier_1_passages'
TIER_2_PASSAGES = 'tier_2_passages'

PASSAGE = 'passage'
TITLE = 'title'
CORPUS_ID = 'corpus_id'
SCORE = 'score'

LLM_PROMPT = 'llm_prompt'
LLM_RESPONSE = 'llm_response'


def embed(encoder, strings):
    question_embeddings = encoder.encode(strings, normalize_embeddings=True, convert_to_tensor=True).half()
    return question_embeddings


def load_embeddings(embeddings_path):
    files = glob.glob(embeddings_path)
    f_to_int = lambda x : int(re.findall(r'\d+', x)[0])
    files = sorted(files, key=f_to_int)
    print(f"{len(files)} embedding shards found {files[0]} ... {files[-1]}")

    # For a larger gpu memory increase the batch size
    # embeddings are sharded into chunks of 128k
    # each shard takes ~100MB in memory
    batch_size = 20
    num_steps = len(files) // batch_size + 1
    print("Num steps", num_steps)

    for batch_idx in range(num_steps):
        fs = files[batch_idx*batch_size: batch_size*(batch_idx+1)]

        corpus_embeddings = np.vstack([np.load(f) for f in fs])
        corpus_embeddings = torch.from_numpy(corpus_embeddings).half().cuda()
        yield corpus_embeddings


def get_encoder(model_dir):
    encoder = SentenceTransformer(model_dir)
    encoder.max_seq_length = 512
    return encoder


def retrieve(samples, embeddings_npy, passages_jsonl, model_dir, top_k=5):
    """
    samples : list of dicts. Must have keys ABCDE and prompt
    """
    encoder = get_encoder(model_dir)
    # Retrieval. Gets sections of wikipedia articles
    # These will be split and reranked later
    for sample in tqdm(samples, desc="preparing queries ... "):
        # We retrieve docs that match very well with either of:
        # 1. just question
        # 2. question + one answer
        # 3. question + all answers

        only_question = ["Represent this sentence for searching relevant passages:" + sample['prompt']]
        questions_with_one_answer = [f"{sample[PROMPT]} {sample[c]}" for c in ABCDE]
        question_with_all_answers = [f"{sample[PROMPT]}" + " ".join([sample[c] for c in ABCDE])]
        questions = questions_with_one_answer \
                    + question_with_all_answers \
                    + only_question

        question_embeddings = embed(encoder, questions)
        sample[TIER_1_QUERIES] = questions
        sample[TIER_1_QUESTION_EMBEDDINGS] = question_embeddings

    offset = 0
    for corpus_embeddings_shard in tqdm(load_embeddings(embeddings_npy), desc="retrieving ..."):
        for sample in samples:
            question_embeddings = sample[TIER_1_QUESTION_EMBEDDINGS]
            # we use only top result,
            # but let's retrieve 5 for debugging
            tier_1_hits = util.semantic_search(question_embeddings,
                                        corpus_embeddings_shard,
                                        top_k=top_k,
                                        score_function=util.dot_score)

            for hits in tier_1_hits:
                for hit in hits:
                    hit[CORPUS_ID] += offset

            # Merge with previous hits and keep top_k
            if TIER_1_HITS in sample:
                for idx, hits in enumerate(tier_1_hits):
                    hits = tier_1_hits[idx] + sample[TIER_1_HITS][idx]
                    hits = sorted(hits, key=lambda x: x[SCORE], reverse=True)[:top_k]
                    tier_1_hits[idx] = hits

            sample[TIER_1_HITS] = tier_1_hits

        offset += len(corpus_embeddings_shard)

    # get all ids that were hit
    # we will load only these passages
    hitids = set()
    for sample in samples:
        tier_1_hits = sample[TIER_1_HITS]
        ids = [h[CORPUS_ID] for hits in tier_1_hits for h in hits]
        hitids.update(ids)

    # get text of passages
    passages = {}
    with open(passages_jsonl) as f:
        # json parsing is costly
        # so we use line number as a proxy for id
        idx = 0
        for line in tqdm(f, "Loading text of matching paragraphs ..."):
            if idx in hitids:
                line = json.loads(line.strip())
                passages[idx] = line
            idx += 1

    # add text to samples
    for sample in tqdm(samples, 'adding text to samples ...'):
        tier_1_hits = sample[TIER_1_HITS]
        sample_passages = []
        # there are multiple hits per sample
        # iterate through each and store each hits passages separately.(for easy debug)
        for hits in tier_1_hits:
            ids = [h[CORPUS_ID] for h in hits]
            q_passages = [passages[_id] for _id in ids]
            sample_passages.append(q_passages)

        sample[TIER_1_PASSAGES] = sample_passages

    for sample in samples:
        del sample[TIER_1_QUESTION_EMBEDDINGS]

    # TODO : This is big
    # passages are repeated across hits
    return samples


def rerank(samples, model_dir):
    # rerank model can be different than retrieval model
    encoder = get_encoder(model_dir)

    # This is a post retrieval stage
    # each sample should have TIER_1 retrieval results
    for sample in tqdm(samples, "reranking ... "):
        questions = sample[TIER_1_QUERIES]
        passages = sample[TIER_1_PASSAGES]

        # Wikipedia has awesome chunking
        # Tier 1 retrieves full sections
        # In Tier 2 we split those sections into paragraphs
        # Then rerank them. This gives us a very small and concise paragraph of context
        # Here we split the sections
        split_passages = []
        split_titles = []
        top_passages = {}
        for q_passages in passages:
            for top_passage in q_passages[0:2]:
                top_passages[top_passage['idx']] = top_passage

        for top_passage in top_passages.values():
            splits = top_passage[PASSAGE].split(".\n")
            split_passages.extend(splits)
            split_titles.extend([top_passage[TITLE]]*len(splits))

        # print(len(split_passages), len(questions))

        # Now that we have split the sections into paragraphs
        # Score each paragraph as before against every question
        # For simplicity questions are kept same as retr. but can be different

        passage_embeddings = embed(encoder=encoder, strings=split_passages)
        question_embeddings = embed(encoder=encoder, strings=questions)
        scores = question_embeddings @ passage_embeddings.T
        scores = scores.detach().cpu().numpy()

        ranks = np.argsort(-scores, axis=1)[:, 0:1]
        ranks = ranks.tolist()
        ranks = [i for r in ranks for i in r]
        ranks = set(ranks)

        chosen_splits = [{PASSAGE: split_passages[i], TITLE: split_titles[i]} for i in ranks]
        sample[TIER_2_PASSAGES] = chosen_splits

    return samples


def retrieve_and_rerank(samples, tmp_dir):
    samples = retrieve(samples=samples,
             model_dir="data/allmodels/model/bge-small-en",
             passages_jsonl="data/deepindex_all_paras/passages.jsonl",
             embeddings_npy="data/deepindex_all_paras/*.npy")

    json.dump(samples, open(f"{tmp_dir}/retrieval.json", 'w'), indent=True)

    samples = json.load(open(f"{tmp_dir}/retrieval.json"))
    samples = rerank(samples, model_dir="data/allmodels/model/bge-small-en")
    json.dump(samples, open(f"{tmp_dir}/reranked.json", 'w'), indent=True)

    return samples


def create_question_prompt(sample, add_unsure=False):
    prompt = "Based on above information answer the following question:"
    question = f"Question : {sample[PROMPT]}"
    answers = [f"\t{c}: {sample[c]}" for c in ABCDE]

    if add_unsure:
        answers.append("\tF: More information is required to answer this question correctly")

    answers = "\n".join(answers)
    qprompt = f"{prompt}\n{question}\n{answers}"
    return qprompt


def vicuna_prompt(prompt, expertise='science and technology'):
    user = f"""You are an expert in {expertise}. Help the user answer their question.\n\nUSER: {prompt}\n"""
    assistant = """ASSISTANT: The correct choice in json format is:""" + '{"correct choice":"'
    return user + assistant


def tournament(all_choices):
    choices = []
    for rank in range(5):
        current = [c[rank] for c in all_choices]
        current = sorted(current, key=lambda x:x[0], reverse=True)
        choices.extend(current)

    winners = []
    for c in choices:
        c = c[1]
        if c not in winners:
            winners.append(c)

    return winners

def is_bad_passage(passage):
    words = passage.split(" ")
    lines = passage.split("\n")
    n_words_per_line = len(words) / len(lines)
    isbad = (len(lines) >= 5) and (n_words_per_line <= 5)
    # if isbad:
    #     print("..............")
    #     maxlen = max([len(line.split(" ")) for line in lines])
    #     print(len(words))
    #     print(len(lines))
    return isbad

def make_prompt(sample):
    passages = sample[TIER_2_PASSAGES]

    experts = {}
    for passage in passages:
        if is_bad_passage(passage[PASSAGE]):
            continue
        title = passage[TITLE]
        experts[title] = experts.get(title, []) + [passage[PASSAGE]]

    openbook = "Here is some wikipedia excerpts that may help you answer:\n"
    for idx, expertise in enumerate(experts):
        information = experts[expertise]
        information = ".".join(information)
        openbook += f"""{idx}. {expertise}: {information}\n"""

    expertise = ",".join(list(experts.keys()))

    p = create_question_prompt(sample, add_unsure=False)
    p = f"{openbook}\n{p}"
    p = vicuna_prompt(prompt=p, expertise=expertise)
    return p


def inverse_permutation(permutation):
    n = len(permutation)
    inverse = [0] * n

    for i in range(n):
        inverse[permutation[i]] = i

    return inverse


def permute(sample, order):
    correct_answer = sample[sample[ANSWER]]
    answers = [sample[a] for a in ABCDE]
    new_answers = [answers[i] for i in order]
    new_correct_answer = ABCDE[new_answers.index(correct_answer)]

    for i, c in enumerate(ABCDE):
        sample[c] = new_answers[i]

    sample[ANSWER] = new_correct_answer
    return sample

def apply_permutation(l, p):
    ln = [l[i] for i in p]
    return ln

def make_choices(samples):
    from llm import GPTQLLM, GGMLLLM
    from train_utils import apk

    llm = GPTQLLM("data/allmodels/model/vicuna13b16k432/")

    apks_3 = []
    apks_1 = []

    for sample in tqdm(samples):
        # print("===========================")
        answer_scores = {}
        for run in range(5):
            # print("-----")
            # Create a shift permutation by rotating the array
            order = list(range(len(ABCDE)))
            order = order[run:] + order[:run]
            inverse_order = inverse_permutation(order)

            # What has abcde become now ?
            new_abcde = apply_permutation(ABCDE, order)
            mapping = dict(zip(ABCDE, new_abcde))

            # apply the permutation to sample
            sample = permute(sample, order)
            prompt = make_prompt(sample)
            # print(prompt)
            choices = llm.process(prompt)
            torch.cuda.empty_cache()

            # if sample[ANSWER] != choices[0][1]:
            #     print(prompt)
            #     print(choices)
            #     print(sample[ANSWER])

            # reverse the permutation
            # collate the scores, against original
            sample = permute(sample, inverse_order)
            for choice in choices:
                c = choice[1]
                s = choice[0]
                real_c = mapping[c]
                answer_scores[real_c] = answer_scores.get(real_c, 0) + s

        # who's the best ?
        choices = sorted(answer_scores, key=answer_scores.get, reverse=True)

        score_1 = apk([sample[ANSWER]], choices, k=1)
        apks_1.append(score_1)

        score_3 = apk([sample[ANSWER]], choices, k=3)
        apks_3.append(score_3)

        # print(choices, score_1, score_3)

    print(sum(apks_1) / len(apks_1))
    print(sum(apks_3) / len(apks_3))

data_df = pd.read_csv("data/extra/openai_questions_kaggle/6000_wiki_en_sci_questions_with_excerpts.csv")

# data_df = pd.read_csv("data/kaggle-llm-science-exam/train.csv")
# data = data_df.to_dict(orient='records')
# data = data[0:200]
#
tmp_dir = "working_6k"
#
# data = retrieve_and_rerank(data, tmp_dir=tmp_dir)
data = json.load(open(f"{tmp_dir}/reranked.json"))
data = data[0:200]
make_choices(data)