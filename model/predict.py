import sys
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
sys.path.append('..')
from config import root_path
from preprocess.data import SelfDataset,id2token
import torch


def predict(path):
    data = SelfDataset(path)
    model = torch.load('new_model10.pkl')
    test = "18270104763河北省唐山市路北区机场路街道华岩北路38号唐山学院南校区"
    test = torch.tensor([data.word_vocab.get(i, '20940') for i in test])
    predicts = []
    true_labels = []
    for tokens, tags in zip(data.word_ids, data.label_ids):
        res = model(tokens)[1]
        predicts.append(res)
        true_labels.append(tags.numpy())
    y_pred = MultiLabelBinarizer().fit_transform(predicts)
    y_true = MultiLabelBinarizer().fit_transform(true_labels)
    print(f1_score(y_true, y_pred, average='macro'))

    id_tokens = id2token('/Users/jgl/Desktop/NLP_Projects/GRU_CRF/data/word.dic')
    id_tag = id2token('/Users/jgl/Desktop/NLP_Projects/GRU_CRF/data/tag.dic')
    # sentences = [id_tokens.get(i.item()) for i in test]
    # print(sentences)
    res = model(test)
    print([id_tag.get(i) for i in res[1]])


if __name__ == '__main__':
    path = '/Users/jgl/Desktop/NLP_Projects/GRU_CRF/data/test.txt'
    predict(path)

