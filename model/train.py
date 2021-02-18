import torch
import  torch.nn as nn
from paddlenlp.data import Stack, Pad,Tuple
from torch.utils.data import Dataset, DataLoader
import sys
import os
from model import BiGRU_CRF
sys.path.append('..')
from preprocess.data import SelfDataset, load_dict
from config import root_path, emb_size, hidden_size, device


def collate_fn(batch):
    """
    动态padding, batch为一部分sample
    :param batch:
    :return:
    """
    def padding(indice, max_length=50, is_words=True):
        """
        pad 函数
        注意 token_type_id 右侧pad是添加1是添加0，1表示属于句子B
        :param indice:
        :param max_length:
        :param pad_idx:
        :return:
        """
        pad_indice = [item for item in indice]
        return torch.tensor(pad_indice, dtype=torch.long)
    word_ids = [data["word_ids"] for data in batch]
    len1 = [data["len"] for data in batch]
    label_ids = [data["label_ids"] for data in batch]
    max_length = max(len1)
    word_ids = padding(word_ids, max_length, True)
    label_ids = padding(label_ids, max_length, False)

    return word_ids, len1, label_ids


class Train:
    def __init__(self):
        print("加载训练数据")
        self.train_ds = SelfDataset(os.path.join(root_path, 'data/train.txt'))
        print("加载测试数据")
        self.dev_ds = SelfDataset(os.path.join(root_path, 'data/dev.txt'))
        # tag_ix = load_dict(os.path.join(root_path, 'data/tag.dic'))
        # self.trainloder = DataLoader(train_ds, batch_size=200, shuffle=True, collate_fn=collate_fn)
        # self.devloder = DataLoader(dev_ds, batch_size=200, shuffle=True, collate_fn=collate_fn)
        self.model = BiGRU_CRF(self.train_ds.word_num, self.train_ds.label_vocab, emb_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in zip(self.train_ds.word_ids+self.dev_ds.word_ids, self.train_ds.label_ids + self.dev_ds.label_ids):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            loss = self.model.neg_log_likelihood(inputs, targets)

            loss.backward()
            self.optimizer.step()
            print('the loss is:%.3f'%loss.item())

            train_loss += loss.item()
        print('Loss: %.3f '%train_loss)


if __name__ == '__main__':
    tr = Train()
    for i in range(10):
        print("the epoch is %d"%(i+1))
        tr.train()
    torch.save(tr.model, 'new_model10.pkl')



