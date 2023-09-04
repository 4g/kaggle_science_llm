from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          LlamaForSequenceClassification,
                          AutoModelForSequenceClassification,
LlamaForCausalLM,
                          BitsAndBytesConfig,
                            LlamaConfig,
                          LlamaModel)

import torch 
from datasets import load_dataset, load_metric


# model_path = "TheBloke/Llama-2-7b-chat-fp16"
model_path = "TheBloke/vicuna-13B-v1.5-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


imdb = load_dataset("classifier_dataset")

# tokenized_imdb = imdb.map(preprocess_function, batched=True)

# LlamaForSequenceClassification

model = LlamaForSequenceClassification.from_pretrained(model_path, device_map=0, torch_dtype=torch.bfloat16)

print(model.config)
from tqdm import tqdm

embeddings = []

shard = 'train'

for elem in tqdm(imdb[shard]):
    input_ids = tokenizer(elem['text'], return_tensors='pt').input_ids.cuda()
    output = model.model.forward(input_ids=input_ids, return_dict=False)
    last_token_embedding = output[0][0, -1, :]
    embedding = last_token_embedding.cpu().detach().numpy()
    embeddings.append(embedding)

import numpy as np
embeddings = np.asarray(embeddings, dtype=np.float32)
np.save(f'{shard}_embeddings.npy', embeddings)
