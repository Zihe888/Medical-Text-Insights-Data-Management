# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for building model. 
@All Right Reserve
'''

import torch
import torch.nn as nn
from transformers import BertModel
#from torchcrf import CRF
from crf import CRF

class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese',return_dict=False)#.to('cpu')
        # self.bert = BertModel.from_pretrained('bert-base-chinese', mirror='tuna')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        #这里改啦
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        # self.linear_BERTCRF = nn.Linear(embedding_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence):
        
        ## BERT + BiLSTM + CRF ##
        #print(sentence)
        #print("-----")
        with torch.no_grad():
            embeds, _  = self.bert(sentence)
            #print(embeds)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        '''
        ## BERT + CRF ##
        with torch.no_grad():
            embeds, _  = self.bert(sentence)
        enc = self.dropout(embeds)
        feats = self.linear_BERTCRF(enc)
        '''
        return feats
    '''
    def forward(self, sentence, tags, mask):
        emissions = self._get_features(sentence)
        loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
        #print("-----loss-----")
        #print(loss)
        #print(loss.shape)
        return loss
    '''    
    def forward(self, sentence,mask):
        emissions = self._get_features(sentence)
        decode=self.crf.decode(emissions,mask)
        return decode
    
    '''
        def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test: # Training，return loss
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            #print("-----loss-----")
            #print(loss)
            #print(loss.shape)
            return loss
        else: # Testing，return decoding
            decode=self.crf.decode(emissions, mask)
            #print("-----decode-----")
            #print(decode)
            return decode
    '''