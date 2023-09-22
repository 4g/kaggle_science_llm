import glob
import math
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import re
from sentence_transformers import SentenceTransformer, util

torch.set_default_device('cuda:0')

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
    num_steps = int(np.ceil(len(files)/batch_size))
    print("Num steps", num_steps)

    for batch_idx in range(num_steps):
        fs = files[batch_idx*batch_size: batch_size*(batch_idx+1)]
        corpus_embeddings = np.vstack([np.load(f) for f in fs])
        corpus_embeddings = torch.from_numpy(corpus_embeddings).half().cuda()
        yield corpus_embeddings


def get_encoder(model_dir):
    encoder = SentenceTransformer(model_dir, device='cuda:0')
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
    import gzip
    with gzip.open(passages_jsonl) as f:
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

# -*- coding: utf-8 -*-
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def rerank(samples, model_dir):
    # rerank model can be different than retrieval model
    encoder = get_encoder(model_dir)
    from collections import Counter
    lengths = Counter()
    divide_array = lambda arr, n: [arr[i:i + len(arr) // n + (1 if i < len(arr) % n else 0)] for i in range(0, len(arr), len(arr) // n + (1 if len(arr) % n > 0 else 0))]

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
            extra_splits = []
            for idx, split in enumerate(splits):
                words = split.split()

                if len(words) > 128:
                    n_splits = math.ceil(len(words) / 128)
                    sentences = split_into_sentences(split)
                    sent_splits = divide_array(sentences, n_splits)
                    sent_splits = [" ".join(s) for s in sent_splits]
                    extra_splits.extend(sent_splits)
                else:
                    extra_splits.append(split)

            split_passages.extend(extra_splits)
            split_titles.extend([top_passage[TITLE]] * len(extra_splits))

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
             passages_jsonl="data/deepindex_all_paras/passages.jsonl.gz",
             embeddings_npy="data/deepindex_all_paras/*.npy")

    json.dump(samples, open(f"{tmp_dir}/retrieval.json", 'w'), indent=True)

    samples = json.load(open(f"{tmp_dir}/retrieval.json"))
    samples = rerank(samples, model_dir="data/allmodels/model/bge-small-en")
    json.dump(samples, open(f"{tmp_dir}/reranked.json", 'w'), indent=True)

    return samples