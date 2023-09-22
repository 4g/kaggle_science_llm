import json
import os

import pandas as pd
from datasets import load_dataset

import numpy as np
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from retrieval import retrieve_and_rerank

ABCDE = "ABCDE"

model_path = "deberta_ft/"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def make_dataset(reranked_json, dataset_path):
    columns = ['prompt', 'A', 'C', 'B', 'D', 'E', 'answer', 'tier_2_passages']

    validation = [reranked_json]

    val_dfs = [pd.DataFrame.from_records(json.load(open(f))) for f in validation]
    val_df = pd.concat(val_dfs)
    val_df = val_df[columns]
    val_df.to_csv(dataset_path, index=False)


def preprocess_function(examples):
    ending_names = [str(i) for i in range(5)]
    first_sentences = [[context] * 5 for context in examples["context"]]

    question_headers = examples["prompt"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i: i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


def make_labels(example):
    answer = example['answer']
    example['label'] = ABCDE.index(answer)
    for i in range(5):
        example[str(i)] = example[ABCDE[i]]
    return example


def is_bad_passage(passage):
    words = passage.split(" ")
    lines = passage.split("\n")
    n_words_per_line = len(words) / len(lines)
    isbad = (len(lines) >= 5) and (n_words_per_line <= 5)
    if passage.count("|") > 10:
        isbad = True
    return isbad


def make_context(example):
    contexts = example["tier_2_passages"]
    contexts = eval(contexts)
    openbook = ""
    max_openbook_len = 1024
    for context in contexts:
        tokens = tokenizer.encode(openbook)
        if len(tokens) > max_openbook_len:
            break
        passage = context['passage']
        if is_bad_passage(passage):
            continue
        title = context['title']
        passage = passage.replace("\n", " ")
        lpassage = len(tokenizer.encode(passage))
        if lpassage > 512:
            print(passage)
            continue
        if lpassage + len(tokens) > max_openbook_len:
            continue
        openbook += f"""{title}: {passage}\n"""

    example['context'] = openbook
    return example

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.tolist()
    labels = eval_pred.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


def predict(dataset):
    model = AutoModelForMultipleChoice.from_pretrained(model_path, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
        output_dir="deberta_ft",
        save_strategy="epoch",
        # optim='adamw_bnb_8bit',
        # max_grad_norm=0.3,
        warmup_ratio=0.03,
        # load_best_model_at_end=True,
        gradient_checkpointing=True,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        logging_steps=50,
        eval_steps=100,
        evaluation_strategy='steps',
        max_steps=12020,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        push_to_hub=False,
        fp16=True,
        tf32=True,
        report_to=["none"],
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    test_predictions = trainer.predict(dataset["validation"]).predictions
    predictions_as_ids = np.argsort(-test_predictions, 1)
    predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
    return predictions_as_answer_letters

tmp_dir = "working"
dataset_path = f"{tmp_dir}/dataset/"
os.makedirs(dataset_path, exist_ok=True)

data_df = pd.read_csv("data/kaggle-llm-science-exam/train.csv")
data_df.fillna("None of the above", inplace=True)
data = data_df.to_dict(orient='records')

data = retrieve_and_rerank(data, tmp_dir=tmp_dir)

# make_dataset(reranked_json=f"{tmp_dir}/reranked.json",
#                        dataset_path=f"{dataset_path}/validation.csv")
#
# dataset = load_dataset(dataset_path)
# dataset = dataset.map(make_context)
# dataset = dataset.map(make_labels)
# dataset = dataset.map(preprocess_function, batched=True)
# results = predict(dataset)
#
# map3 = 0
# map1 = 0
# from train_utils import apk
# for total, (elem, result) in enumerate(zip(data, results)):
#     map1 += apk([elem['answer']], result, k=1)
#     map3 += apk([elem['answer']], result, k=3)
#     print(map1/(total+1), map3/(total+1))
