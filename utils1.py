# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for implementing Dataset. 
@All Right Reserve
'''

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

bert_model = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-BODY','I-TEST', 'I-EXAMINATIONS',
         'I-TREATMENT', 'B-DRUG', 'B-TREATMENT', 'I-DISEASES', 'B-EXAMINATIONS', 
         'I-BODY', 'B-TEST', 'B-DISEASES', 'I-DRUG', 'E-BODY', 'E-DISEASES',
         'E-DRUG', 'E-EXAMINATIONS', 'E-TEST', 'E-TREATMENT', 'S-BODY', 'S-DISEASES',
         'S-DRUG', 'S-EXAMINATIONS', 'S-TEST', 'S-TREATMENT')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2

class NerDataset(Dataset):
    ''' Generate our dataset '''
    def __init__(self, f_path):
        self.sents = []
        #self.tags_li = []

        with open(f_path, 'r', encoding='utf-8') as f:
            # print(f.readlines())
            lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip())!=0]
            
          
        #tags =  [line.split('\t')[1] for line in lines]
        words = [line.split('\t')[0] for line in lines]

        word = []
        for char in words:
            if char != '。':
                word.append(char)
            else:
                if len(word) > MAX_LEN:
                    self.sents.append(['[CLS]'] + word[:MAX_LEN] + ['[SEP]'])
                else:
                    self.sents.append(['[CLS]'] + word + ['[SEP]'])
                word = []

    def __getitem__(self, idx):
        words = self.sents[idx]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        return token_ids

    def __len__(self):
        return len(self.sents)

def PadBatch(batch):
    maxlen = max([len(i) for i in batch])
    #print(type(maxlen))
    #print(type(batch[0]))
    token_tensors = torch.LongTensor([i + [0] * (maxlen - len(i)) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, mask

    '''
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, mask
    '''