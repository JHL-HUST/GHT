import pickle
from dataset import BaseDataset
import numpy as np
import os
from tqdm import tqdm


data_path = os.path.join('data', 'ICEWS0515')
trainpath = os.path.join(data_path, 'train.txt')
validpath = os.path.join(data_path, 'valid.txt')
testpath = os.path.join(data_path, 'test.txt')
statpath = os.path.join(data_path, 'stat.txt')
baseDataset = BaseDataset(trainpath, testpath, statpath, validpath)

trainQuadruples = baseDataset.get_reverse_quadruples_array(baseDataset.trainQuadruples, baseDataset.num_r)

relations_scores = {i: [0, [0 for j in range(baseDataset.num_r * 2)]] for i in range(baseDataset.num_r * 2)}

for i in tqdm(range(len(trainQuadruples))):
    quad = trainQuadruples[i]
    history = trainQuadruples[trainQuadruples[:, 3] < quad[3]]
    if len(history) <= 0:
        continue
    body = history[(history[:, 0] == quad[0]) & (history[:, 2] == quad[2])][:, 1]
    reverse_body = history[(history[:, 0] == quad[2]) & (history[:, 2] == quad[0])][:, 1]

    all_body = np.unique(np.concatenate([body, reverse_body]))

    if len(all_body) <= 0:
        continue

    relations_scores[quad[1]][0] += 1
    for rel in all_body:
        relations_scores[quad[1]][1][rel] += 1


conf = np.zeros([baseDataset.num_r * 2, baseDataset.num_r * 2])
for k, v in relations_scores.items():
    if v[0] != 0:
        conf[k] = np.array(v[1]) / v[0]

print(conf)
print(conf[0])
pickle.dump(conf, open(os.path.join(data_path, 'conf.pkl'), 'wb'))