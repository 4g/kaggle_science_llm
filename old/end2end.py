import copy

from sentence_transformers import SentenceTransformer, CrossEncoder, util

from tqdm import tqdm
import pandas as pd
import random

import json
import numpy as np
import torch
from scipy.special import log_softmax

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True

BASE_PATH = "data"

INPUT_BASE_PATH = f"{BASE_PATH}/kaggle-llm-science-exam"
MODELS_PATH = f"{BASE_PATH}/allmodels/model"
DEEPINDEX_PATH = f"{BASE_PATH}/deepindex_science_paras"

RELEVANT_JSON_PATH = "relevant_passages.json"
SUBMISSION_PATH = "submission.csv"

TEST_CSV = f"{INPUT_BASE_PATH}/test.csv"
TRAIN_CSV = f"{INPUT_BASE_PATH}/train.csv"

# TEST_CSV = f"{BASE_PATH}/extra/openai_questions_kaggle/6000_all_categories_questions.csv"
#
TEST_CSV = f"{BASE_PATH}/extra/openai_questions_kaggle/6000_wiki_en_sci_questions_with_excerpts.csv"
TRAIN_CSV = TEST_CSV


LLM_GPU = 0

# LLM_PATH = f"{MODELS_PATH}/vicuna13b4128"
# LLM_PATH = "TheBloke/WizardLM-13B-V1.2-GPTQ"
# LLM_PATH = "TheBloke/OpenChat_v3.2-GPTQ"
# LLM_PATH = "TheBloke/vicuna-7B-v1.5-16K-GPTQ"
# LLM_PATH = "TheBloke/vicuna-13B-v1.5-GPTQ"
LLM_PATH = f"{MODELS_PATH}/vicuna13b16k432"

TOP_PARAS = 1000
N_RUNS = 1
N_TOKENS = 100000
TOP_RETRIEVED = 10

schema = {
    "type": "object",
    "properties": {
        "correct choice": {
            "type": "string",
        },
    }
}

schema = json.dumps(schema)

random.seed(31)
abcde = ['A', 'B', 'C', 'D', 'E']
ORDERS = []
for i in range(N_RUNS):
    ORDERS.append(copy.deepcopy(abcde))
    random.shuffle(abcde)

# ORDERS = [
#     ['A', 'B', 'C', 'D', 'E'],
#     ['B', 'C', 'D', 'E', 'A'],
#     ['C', 'D', 'E', 'A', 'B'],
#     ['D', 'E', 'A', 'B', 'C'],
#     ['E', 'A', 'B', 'C', 'D']
# ][:N_RUNS]

class DeepIndex:
    def __init__(self):
        self.bi_encoder = SentenceTransformer(f'{MODELS_PATH}/bge-small-en')
        self.bi_encoder.max_seq_length = 512

        self.corpus_embeddings = None
        self.index_dir = f'{DEEPINDEX_PATH}'

    def search(self, query, top_k=256, is_question=True):
        with (torch.inference_mode(),
                torch.cuda.amp.autocast(enabled=True)):

            if is_question:
                query = "Represent this sentence for searching relevant passages:" + query

            question_embedding = self.bi_encoder.encode(query,
                                                        convert_to_tensor=True,
                                                        show_progress_bar=False,
                                                        normalize_embeddings=True)

            question_embedding = question_embedding.half().cuda()
            hits = util.semantic_search(question_embedding,
                                        self.corpus_embeddings,
                                        top_k=top_k,
                                        score_function=util.dot_score)
            hits = hits[0]

        return hits

    def embed(self, queries):
        with (torch.inference_mode(),
              torch.cuda.amp.autocast(enabled=True)):

            embeddings = self.bi_encoder.encode(queries,
                                                convert_to_tensor=True,
                                                show_progress_bar=False,
                                                normalize_embeddings=True)

            return embeddings

    def hits_to_passages(self, hitids):
        ids_needed = set()
        needed_passages = {}

        for h in hitids:
            ids_needed.update(h)
        idx = 0
        # print(ids_needed)
        with open(f'{self.index_dir}/passages.jsonl') as f:
            for line in tqdm(f, "Loading text of matching paragraphs"):
                if idx in ids_needed:
                    line = json.loads(line.strip())
                    needed_passages[idx] = line
                idx += 1

        return needed_passages

    def load_index(self, index_dir):
        self.corpus_embeddings = np.load(f'{index_dir}/corpus_embeddings.npy')
        self.corpus_embeddings = torch.from_numpy(self.corpus_embeddings).half().cuda()



def shuffle_answers(sample, new_order):
    old_order = ['A', 'B', 'C', 'D', 'E']
    new_sample = {new_order[old_order.index(i)]: sample[i] for i in old_order}
    order_map = {new_order[old_order.index(i)]: i for i in old_order}
    return new_sample, order_map

# def create_prompt(passages, question, answers):
#     openbook = " ".join([p['passage'] for p in passages])
#     prompt = f"""{openbook}\n\
# Based on above information answer the following question:\n\
# Question : {question}\n\
#     A: {answers['A']}\n\
#     B: {answers['B']}\n\
#     C: {answers['C']}\n\
#     D: {answers['D']}\n\
#     E: {answers['E']}\n\
# """
#     llmprompt = f"""A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input\n\nUSER: {prompt}\n{schema}\nASSISTANT: Sure, Here is the correct choice in JSON schema format:""" + "{'correct choice':'"
#     return llmprompt
#
def create_prompt(passages, question, answers):
    openbook = " ".join([p['passage'] for p in passages])
    title = ",".join(set([p['title'] for p in passages]))
    openbook = openbook.strip()
    title = title.strip()

    prompt = f"""{openbook}\n\
Based on above information answer the following question:\n\
Question : {question}\n\
    A: {answers['A']}\n\
    B: {answers['B']}\n\
    C: {answers['C']}\n\
    D: {answers['D']}\n\
    E: {answers['E']}\n\
"""
    llmprompt = f"""You are an expert in {title}. Help the user answer their question.\n\nUSER: {prompt}\n\
ASSISTANT: The correct choice in json format is:""" + '{"correct choice":"'

    return llmprompt

def get_answers(sample, passages, llm):
    scores = {}
    orders = ORDERS
    for i in range(len(orders)):
        new_sample, order_map = shuffle_answers(sample, orders[i])
        llmprompt = create_prompt(passages,
                                  question=sample.prompt,
                                  answers=new_sample)
        # print(llmprompt)

        ans = llm.process(llmprompt)
        torch.cuda.empty_cache()
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


def create_submission(data_df, relevant_paras_path):
    llm = GPTQLLM(LLM_PATH)
    relevant_paras = []
    for line in open(relevant_paras_path):
        para = json.loads(line.strip())
        relevant_paras.append(para)

    pred_answers = []
    top_paras = TOP_PARAS
    for idx in tqdm(range(len(data_df))):
        example = data_df.iloc[idx]

        # openbook = example['wikipedia_excerpt']
        # openbook = make_openbook(relevant_paras[idx], top_paras)

        # print(openbook)

        out = get_answers(example, relevant_paras[idx], llm)
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

def main():

    # make_index()
    n_rows = 200
    relevant_json_path = RELEVANT_JSON_PATH
    data_df = pd.read_csv(TEST_CSV).head(n_rows)
    # #
    get_relevant_sentences(data_df=data_df,
                           deepindex_dir=DEEPINDEX_PATH,
                           relevant_paras_path=relevant_json_path)


    data_df = create_submission(data_df=data_df, relevant_paras_path=relevant_json_path)

    # data_df = data_df[['id', 'prediction']]
    data_df.to_csv('submission.csv', index=False)

    # data_df = pd.read_csv("submission.csv")
    train_df = pd.read_csv(TRAIN_CSV).head(n_rows)
    score_results(data_df, train_df, k=1)
    score_results(data_df, train_df, k=3)


if __name__ == "__main__":
    main()
