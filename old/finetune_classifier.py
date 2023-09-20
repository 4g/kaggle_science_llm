import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras_core as keras
import numpy as np
from datasets import load_dataset

def shuffle_answers(sample):
    order = ['A', 'B', 'C', 'D', 'E']
    old_answer_index = order.index(sample['answer'])
    permutation = list(range(len(order)))
    random.shuffle(permutation)
    new_answer_index = permutation.index(old_answer_index)
    new_answer = order[new_answer_index]
    new_sample = sample.copy()

    for idx, key in enumerate(order):
        new_sample[order[permutation.index(idx)]] = sample[key]

    new_sample['answer'] = new_answer
    return new_sample

def create_embeddings(data_df, prefix, augment):
    from end2end import GPTQLLM, LLM_PATH, create_prompt

    llm = GPTQLLM(model_path=LLM_PATH)

    samples = []
    embeddings = []
    pred_answer = []
    epochs = 5

    for epoch in range(epochs):
        data = data_df.to_dict(orient='records')
        for sample in tqdm(data):
            if augment:
                sample = shuffle_answers(sample)

            prompt = create_prompt(sample['wikipedia_excerpt'],
                                   question=sample['prompt'],
                                   answers=sample)
            sample['prompt'] = prompt
            logits = llm.get_logits(prompt, char_head=False)
            embeddings.append(logits)
            samples.append(sample)

    np.save(file=f'{prefix}embeddings.npy', arr=embeddings)
    json.dump(fp=open(f'{prefix}samples.json', 'w'), obj=samples)


def get_test_dataset():
    df = pd.read_csv('data/kaggle-llm-science-exam/train.csv')
    relevant_passages = json.load(open('kaggle_relevant_passages.json'))
    data = df.to_dict(orient='records')
    for idx in range(len(data)):
        elem = data[idx]
        passages = relevant_passages[idx]
        best_passage = passages[0]
        elem['wikipedia_excerpt'] = f"{best_passage['title']}:{best_passage['passage']}"

    df = pd.DataFrame.from_records(data)
    return df

def make_dataset():
    df1 = pd.read_csv('data/extra/openai_questions_kaggle/6000_all_categories_questions_with_excerpts.csv')
    df2 = pd.read_csv('data/extra/openai_questions_kaggle/6000_wiki_en_sci_questions_with_excerpts.csv')
    train_df = pd.concat([df1])
    create_embeddings(train_df, prefix="data/classifier_data/train_hidden_", augment=False)

    # test_df = get_test_dataset()

    # create_embeddings(test_df, prefix="data/classifier_data/test_hidden_", augment=True)



def load_dataset(shard):
    X = np.load(f"data/classifier_data/{shard}_hidden_embeddings.npy")
    samples = json.load(open(f"data/classifier_data/{shard}_hidden_samples.json"))

    labels = []
    abcde = ['A','B','C','D','E']

    for sample in samples:
        label = abcde.index(sample['answer'])
        labels.append(label)

    labels_onehot = np.zeros((len(labels), 5), dtype=np.float32)
    for idx, label in enumerate(labels):
        labels_onehot[idx][label] = 1.0

    return X, labels_onehot


def apk(actual, predicted, k=3):
    assert (len(np.unique(predicted)) == len(predicted))

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def train_classifier():
    X, y = load_dataset('train')
    test_X, test_y = load_dataset('test')

    inp = keras.layers.Input(shape=(5120,))
    inp = keras.layers.UnitNormalization()(inp)
    dropout = keras.layers.Dropout(0.4)(inp)
    dense1 = keras.layers.Dense(5, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform)(dropout)
    model = keras.models.Model(inputs=inp, outputs=dense1)
    # model.trainable = False

    model.summary()
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=.004),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.6),
                  metrics=[keras.metrics.CategoricalAccuracy()])

    model.fit(X, y, batch_size=128, epochs=100, validation_data=(test_X, test_y), shuffle=True)
    y_preds = model.predict(X)
    score = 0
    for idx in range(len(y_preds)):
        y_pred = y_preds[idx]
        y_pred = np.argsort(y_pred)[::-1]
        y_true = np.argsort(y[idx])[::-1][0]
        score += apk([y_true], y_pred, k=3)

    score = score / len(y_preds)
    print(score)
    model.save('test_classifier.keras')

    model = keras.models.load_model('test_classifier.keras')
    model.evaluate(X, y)


# make_dataset()
train_classifier()