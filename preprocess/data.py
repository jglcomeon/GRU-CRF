import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append('..')
from config import root_path


def load_dict(dict_path):
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


def id2token(dict_path):
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[int(value)] = key
    return vocab

def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids


class SelfDataset(Dataset):
    def __init__(self, path):
        self.word_vocab = load_dict(os.path.join(root_path, 'data/word.dic'))
        self.label_vocab = load_dict(os.path.join(root_path, 'data/tag.dic'))
        self.word_ids = []
        self.label_ids = []
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                sub_word_ids = torch.tensor(convert_tokens_to_ids(words, self.word_vocab, 'OOV'), dtype=torch.long)
                sub_label_ids = torch.tensor(convert_tokens_to_ids(labels, self.label_vocab, 'O'), dtype=torch.long)
                self.word_ids.append(sub_word_ids)
                self.label_ids.append(sub_label_ids)
        self.word_num = max(self.word_vocab.values())
        self.label_num = max(self.label_vocab.values())

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, item):
        output = {
            'word_ids': self.word_ids[item],
            'len': len(self.word_ids[item]),
            'label_ids': self.label_ids[item]
        }
        return output


if __name__ == '__main__':
    train_ds = SelfDataset(os.path.join(root_path, 'data/train.txt'))
    dev_ds = SelfDataset(os.path.join(root_path, 'data/dev.txt'))
    test_ds = SelfDataset(os.path.join(root_path, 'data/test.txt'))