from sentence_transformers import SentenceTransformer, CrossEncoder, util

from tqdm import tqdm
import pandas as pd
from random import shuffle

import json
import numpy as np
import torch

BASE_PATH = "data"

INPUT_BASE_PATH = f"{BASE_PATH}/kaggle-llm-science-exam"
MODELS_PATH = f"{BASE_PATH}/allmodels/model"
DEEPINDEX_PATH = f"{BASE_PATH}/deepindex_all"

RELEVANT_JSON_PATH = "relevant_passages.json"
SUBMISSION_PATH = "submission.csv"
TEST_CSV = f"{INPUT_BASE_PATH}/test.csv"
TRAIN_CSV = f"{INPUT_BASE_PATH}/train.csv"
LLM_GPU = 0

LLM_PATH = "TheBloke/vicuna-13B-v1.5-GPTQ"

schema = {
    "type": "object",
    "properties": {
        "correct choice": {
            "type": "string",
        },
    }
}

schema = json.dumps(schema)

TOP_PARAS = 5
N_RUNS = 5
N_TOKENS = 256
TOP_RETRIEVED = 64

class GPTQLLM:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                        revision="gptq-4bit-32g-actorder_True",
                                                        torch_dtype=torch.bfloat16,
                                                        device_map=LLM_GPU)
        self.model.to_bettertransformer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.token_map = self.get_token_map(tokens=None)
        print(self.token_map)
        print("Model Up")

    def get_token_map(self,  tokens):
        if tokens is None:
            tokens = ['A', 'B', 'C', 'D', 'E']
        sent = "," + ",".join(tokens) + ","
        ids = self.tokenizer.encode(sent)
        indices = [i for i in tokens]

        for id in ids:
            token = self.tokenizer.decode(id)
            if token in tokens:
                indices[tokens.index(token)] = id

        return indices, tokens

    def process(self, question):
        # print(question)
        with (torch.inference_mode(),
              torch.amp.autocast(device_type=self.model.device.type),
              torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)):
            input_tensor = self.tokenizer.encode(question, return_tensors="pt")
            input_tensor = input_tensor.cuda()
            output = self.model.forward(input_tensor, use_cache=True)
            logits = output.logits[0, -1]
            indices = self.token_map[0]
            tokens = self.token_map[1]
            scores = logits.detach().cpu().numpy()
            choices = sorted(list(zip(list(scores[indices]), tokens)), key=lambda x: x[0], reverse=True)
        return choices

class DeepIndex:
    def __init__(self):
        self.bi_encoder = SentenceTransformer(f'BAAI/bge-small-en')
        self.bi_encoder.max_seq_length = 512
        self.cross_encoder = CrossEncoder(f'{MODELS_PATH}/ms-marco-MiniLM-L-12-v2')
        self.corpus_embeddings = None
        self.index_dir = f'{DEEPINDEX_PATH}'

    def search(self, query, top_k=256):
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            if self.corpus_embeddings is None:
                raise
            query = "Represent this sentence for searching relevant passages:" + query
            question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
            question_embedding = question_embedding.half().cuda()
            hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=top_k,score_function=util.dot_score)
            hits = hits[0]

        return hits

    def rerank(self, query, passages, key='passage'):
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            cross_inp = [[query, passage[key]] for passage in passages]
            cross_scores = self.cross_encoder.predict(cross_inp, show_progress_bar=False)

        return cross_scores

    def hits_to_passages(self, hitids):
        ids_needed = set()
        needed_passages = {}

        for h in hitids:
            ids_needed.update(h)
        idx = 0
        with open(f'{self.index_dir}/passages.jsonl') as f:
            for line in tqdm(f):
                line = json.loads(line.strip())
                if idx in ids_needed:
                    needed_passages[idx] = line
                idx += 1

        for idx in ids_needed:
            if idx not in needed_passages:
                print(idx)

        return needed_passages

    def search_in(self, query, sentences):
        cross_inp = [[query, hit] for hit in sentences]
        cross_scores = self.cross_encoder.predict(cross_inp, show_progress_bar=False)
        return cross_scores

    def prepare_index(self, index_dir, passages, titles):
        json.dump(passages, open(f'{index_dir}/passages.json', 'w'))
        json.dump(titles, open(f'{index_dir}/titles.json', 'w'))

        corpus_embeddings = self.bi_encoder.encode(passages,
                                                   batch_size=512,
                                                   show_progress_bar=True,
                                                   convert_to_numpy=True)

        np.save(f'{index_dir}/corpus_embeddings.npy', corpus_embeddings)

    def load_index(self, index_dir):
        self.corpus_embeddings = np.load(f'{index_dir}/corpus_embeddings.npy')
        self.corpus_embeddings = torch.from_numpy(self.corpus_embeddings).half().cuda()


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def get_relevant_sentences(data_df, deepindex_dir, relevant_paras_path):
    searcher = DeepIndex()
    searcher.load_index(index_dir=deepindex_dir)

    hitids = []

    print('looking for relevant paragraphs')
    for idx in tqdm(range(len(data_df))):
        example = data_df.iloc[idx]
        query = f"{example.prompt}"
        hits = searcher.search(query, top_k=TOP_RETRIEVED)

        _hitids = [hit['corpus_id'] for hit in hits]
        hitids.append(_hitids)

    print('converting ids to passages')
    passages = searcher.hits_to_passages(hitids)
    expanded_hitids = []
    expanded_passages = {}

    for hits in hitids:
        new_hits = []
        for hit in hits:
            passage = passages[hit]
            # print("========================")
            # print(hit, passage)
            title = passage['title']
            split_passages = passage['passage'].split("\n")
            for i, s in enumerate(split_passages):
                new_hit = f"{hit}_{i}"
                # print("----------------------")
                new_passage = {'passage':s, 'title':title}
                # print(new_hit, new_passage)
                expanded_passages[new_hit] = new_passage
                new_hits.append(new_hit)
        expanded_hitids.append(new_hits)

    hitids = expanded_hitids
    passages = expanded_passages

    relevant_paras = []
    top_k = TOP_PARAS
    print('reranking passages')

    for idx in tqdm(range(len(data_df))):
        example = data_df.iloc[idx]
        query = f"{example.prompt}"
        hits = hitids[idx]
        idx_passages = [passages[i] for i in hits]
        scores = searcher.rerank(query=query, passages=idx_passages, key='passage')
        top_k_index = np.argsort(scores)[::-1][:top_k]
        paras = []
        for i in top_k_index:
            passage = idx_passages[i]
            paras.append(passage)

        relevant_paras.append(paras)

    json.dump(relevant_paras, open(relevant_paras_path, 'w'))
    return relevant_paras


def shuffle_answers(sample, new_order):
    old_order = ['A', 'B', 'C', 'D', 'E']
    new_sample = {new_order[old_order.index(i)]: sample[i] for i in old_order}
    order_map = {new_order[old_order.index(i)]: i for i in old_order}
    return new_sample, order_map

def create_prompt(openbook, question, answers):
    prompt = f"""{openbook}\n\
Based on above information, answer the following question.\n\n\
Question : {question} \n\
    A: {answers['A']}\n\
    B: {answers['B']}\n\
    C: {answers['C']}\n\
    D: {answers['D']}\n\
    E: {answers['E']}\n\
"""
    llmprompt = f"""A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input\n\nUSER: {prompt}\nASSISTANT: Sure, Here is the correct choice in JSON schema format:""" + "{'correct choice':'"
    return llmprompt

def get_answers(sample, openbook, llm, repeat=3):
    scores = {}
    orders = [['A', 'B', 'C', 'D', 'E'],
              ['E', 'D', 'C', 'B', 'A'],
              ['B', 'A', 'C', 'D', 'E'],
              ['D', 'E', 'C', 'B', 'A'],
              ['C', 'A', 'D', 'B', 'E']]

    for i in range(len(orders)):
        new_sample, order_map = shuffle_answers(sample, orders[i])
        llmprompt = create_prompt(openbook,
                                  question=sample.prompt,
                                  answers=new_sample)
        # print(llmprompt)

        ans = llm.process(llmprompt)
        cans = []
        for idx, elem in enumerate(ans):
            x = elem[0], order_map[elem[1]]
            cans.append(x)

        ans = cans
        for s, e in ans:
            scores[e] = scores.get(e, 0) + s

    ans = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ans = [a[::-1] for a in ans]
    return ans

def make_openbook(passages, top_paras=3):
    para_to_text = lambda x: f"{x['title']}:{x['passage']}"
    text = [para_to_text(t) for t in passages[0:top_paras]]
    openbook = " ".join(text)
    openbook = " ".join(openbook.split()[:N_TOKENS])
    return openbook

def create_submission(data_df, relevant_paras_path):
    llm = GPTQLLM(LLM_PATH)

    relevant_paras = json.load(open(relevant_paras_path))
    pred_answers = []
    top_paras = TOP_PARAS

    for idx in tqdm(range(len(data_df))):
        example = data_df.iloc[idx]
        openbook = make_openbook(relevant_paras[idx], top_paras)
        out = get_answers(example, openbook, llm, repeat=N_RUNS)
        pred_answers.append(out)

    data_df["prediction"] = [" ".join([j[1] for j in i][0:3]) for i in pred_answers]
    return data_df


def score_results(data_df, train_df, k=3):
    scores = []

    for i in range(len(data_df)):
        pred_row = data_df.iloc[i]
        actual_row = train_df.iloc[i]
        pred = pred_row.prediction.split()
        actual = actual_row.answer.upper()
        score = apk([actual], pred, k=k)
        scores.append(score)

    print(sum(scores) / len(scores))


def get_documents():
    from pathlib import Path
    science_pages = json.load(open('science_pages.json'))
    science_pages = set(science_pages)

    wikifs = Path('data/wikipedia/graelo_wikipedia/').glob('*.parquet')
    wikifs = list(wikifs)

    passages = []
    titles = []
    for num, f in tqdm(enumerate(wikifs)):
        df = pd.read_parquet(f)
        for i in range(len(df)):
            elem = df.iloc[i]
            title = elem.title
            text = elem.text
            process_title = title.lower().replace(" ", "_")

            if process_title not in science_pages:
                continue

            parts = text.split("\n")
            for part in parts:
                if len(part.split()) > 30:
                    for para in part.split("\n"):
                        passages.append(para)
                        titles.append(title)
        print(len(passages), len(titles))
    return passages, titles


def make_index():
    passages, titles = get_documents()
    searcher = DeepIndex()
    searcher.prepare_index(DEEPINDEX_PATH, passages=passages, titles=titles)

def main():

    # make_index()

    relevant_json_path = RELEVANT_JSON_PATH
    data_df = pd.read_csv(TEST_CSV)

    get_relevant_sentences(data_df=data_df, deepindex_dir=DEEPINDEX_PATH, relevant_paras_path=relevant_json_path)
    data_df = create_submission(data_df=data_df, relevant_paras_path=relevant_json_path)

    data_df = data_df[['id', 'prediction']]
    data_df.to_csv('submission.csv', index=False)

    data_df = pd.read_csv("submission.csv")
    train_df = pd.read_csv(TRAIN_CSV)
    score_results(data_df, train_df, k=1)
    score_results(data_df, train_df, k=3)


if __name__ == "__main__":
    main()
