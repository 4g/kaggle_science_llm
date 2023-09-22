from train_utils import apk
import pandas as pd

submission = pd.read_csv('submission.csv')
train = pd.read_csv('data/kaggle-llm-science-exam/train.csv')

map1 = []
map3 = []
for i in range(200):
    s = submission.iloc[i]
    t = train.iloc[i]
    answer = t['answer']
    preds = s['prediction'].split()
    map3.append(apk([answer], preds, 3))
    map1.append(apk([answer], preds, 1))

print(sum(map1)/len(map1), sum(map3)/len(map3))