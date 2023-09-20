import glob

import torch
torch.set_num_threads(4)

from sentence_transformers import SentenceTransformer, CrossEncoder, util

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import shuffle
import jsonlines

BASE_PATH = "data"
MODELS_PATH = f"{BASE_PATH}/allmodels/model"
DEEPINDEX_PATH = f"{BASE_PATH}/deepindex_all_paras"

def make_passages():
    from pathlib import Path
    science_pages = json.load(open('science_pages.json'))
    science_pages = set(science_pages)

    wikifs = Path('data/wikipedia/graelo_wikipedia/').glob('*.parquet')
    wikifs = list(wikifs)


    titles = []

    index_dir = f'{DEEPINDEX_PATH}'
    fhandle = open(f'{index_dir}/passages.jsonl', 'w')
    n_passages = 0
    for num, f in tqdm(list(enumerate(wikifs))):
        df = pd.read_parquet(f)
        for i in range(len(df)):
            elem = df.iloc[i]
            title = elem.title
            text = elem.text
            process_title = title.lower().replace(" ", "_")

            # if process_title not in science_pages:
            #     continue

            parts = text.split("\n\n")

            passages = []
            passage = []
            maxlen = 512

            # for part in parts:
            #     words = part.strip().split() + ["\n"]
            #     if len(words) < 30:
            #         continue
            #
            #     if len(words) + len(passage) < maxlen:
            #         passage.extend(words)
            #     else:
            #         passages.append(passage)
            #         passage = words

            # if len(passage) > 0:
            #     passages.append(passage)

            passages = parts

            for passage in passages:
                words = passage.replace("\n", " ").split()
                if len(words) < 30:
                    continue
                x = {"passage": passage, "title": title, "idx": n_passages}
                x = json.dumps(x)
                fhandle.write(x + "\n")
                n_passages += 1

        print(n_passages)
    fhandle.close()

def get_passages(passages_path, batch_size=128):
    num_lines = 0
    passages = []
    f = open(passages_path)
    for line in f:
        line = json.loads(line.strip())
        passages.append(line['passage'])
        if len(passages) % batch_size == 0:
            yield passages
            passages = []
    else:
        yield passages

def make_index():
    passages_path = f'{DEEPINDEX_PATH}/passages.jsonl'
    bi_encoder = SentenceTransformer('BAAI/bge-small-en')

    bi_encoder.max_seq_length = 512

    batch_size = 128

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True

    with (torch.inference_mode(),
            torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16),
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)):

        for batch_idx, batch in enumerate(get_passages(passages_path, batch_size=batch_size*1000)):
            print("Batch index", batch_idx)
            batch_embeddings = bi_encoder.encode(batch,
                                                   batch_size=batch_size,
                                                   convert_to_numpy=False,
                                                   convert_to_tensor=True,
                                                   normalize_embeddings=True,
                                                   show_progress_bar=True)

            batch_embeddings = batch_embeddings.detach().cpu().numpy().astype(np.float16)
            np.save(f"{DEEPINDEX_PATH}/embeddings_{batch_idx}.npy", batch_embeddings)

def join_npy():
    files = glob.glob(f"{DEEPINDEX_PATH}/embeddings_*.npy")
    files = sorted(files, key=lambda x:int(x.split(".")[0].split("_")[-1]))
    arrays = []
    for f in files:
        x = np.load(f)
        arrays.append(x)

    arrays = np.vstack(arrays)
    np.save(f"{DEEPINDEX_PATH}/corpus_embeddings.npy", arrays)

def main():
    # make_passages()
    # make_index()
    join_npy()

if __name__ == "__main__":
    main()
