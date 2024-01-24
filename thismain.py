# -*- coding: utf-8 -*-
'''
@Author: Xavier WU
@Date: 2021-11-30
@LastEditTime: 2022-1-6
@Description: This file is for training, validating and testing. 
@All Right Reserve
'''
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import copy
import warnings
import argparse
import numpy as np
from sklearn import metrics
# 这里改啦 0516
#from thismoBi import BiLSTM_CRF
# 8.19
from thismodels import Bert_BiLSTM_CRF
#from BERTmodels import BertME
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(iterator):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        
        # print(x)
        # 8.19
        #loss = model(x, y)
        # 0516
        # 这里改啦
        loss = model(x, y, z)
        # loss = loss[0]
        # print(loss)
        losses += loss.item()
        #print(losses)
        #print("------loss-------")
        """ Gradient Accumulation """
        '''
          full_loss = loss / 2                            # normalize loss 
          full_loss.backward()                            # backward and accumulate gradient
          if step % 2 == 0:             
              optimizer.step()                            # update optimizer
              scheduler.step()                            # update scheduler
              optimizer.zero_grad()                       # clear gradient
        '''
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses/step))

def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    Y_test = []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # iterator是dataloader，返回(inputs,labels)
            # i表示训练完所有数据一次需要的次数
            # batch是(x_batch,y_batch),x_batch和y_batch的长度都是batch size的大小，x中每一个元素都代表一句话
            step += 1

            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            y_hat = model(x, y, z, is_test=True)
            # y_test = torch.squeeze(y_hat)
            # y_test的形状和y一样 
            # y_hat应当是预测的效果
            
            ##8.19
            ##loss = model(x, y)
            # 0516
            # 这里改啦
            loss = model(x, y, z)
            losses += loss.item()
            
            # Save prediction
            y_hat = torch.squeeze(y_hat)
            mask = (z==1)
            y_hat_ = torch.masked_select(y_hat, mask)
            Y_hat.append(y_hat_.cpu())
            
            '''
            for j in y_hat:
                # 0513 这里改啦
                Y_hat.extend(j.cpu())
                # 经过extend之后，Y_hat的长度应当与 max数据长度 * batch_size，相当于把一个batch里面预测的y都拼接成了一条
                #Y_hat.extend(j)            
            '''    
            
            # Save labels       
            mask = (z==1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())

    
    # print(Y_test)
    # print("------Y_hat------")
    # print(Y_hat)
    # print("-----")
    # print(Y)
    # print("-----newY------")
    Y = torch.cat(Y, dim=0).numpy() #拼接Y中不同维度的列表，竖着拼接
    # print(Y)
    Y = np.array(Y)
    Y_hat = torch.cat(Y_hat, dim=0).numpy() 
    Y_hat = np.array(Y_hat)
    # print(len(Y_hat))
    # print(len(Y))
    # Y_hat = np.array(Y_hat.to('cpu'))
    # Y_hat = Y_hat.numpy()
    # acc = np.mean(Y_hat == Y)*100
    # 0513 这里改啦
    acc = (Y_hat == Y).mean()*100
    
    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses/step, acc))
    return model, losses/step, acc
'''
def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1

            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            y_hat = model(x, y, z, is_test=True)

            loss = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (z==1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean()*100

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses/step, acc))
    return model, losses/step, acc
'''
def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            x = x.to(device)
            z = z.to(device)
            y_hat = model(x, y, z, is_test=True)
            
            # Save prediction
            y_hat = torch.squeeze(y_hat)
            mask = (z==1)
            y_hat_ = torch.masked_select(y_hat, mask)
            Y_hat.append(y_hat_.cpu())
            
            # Save labels
            mask = (z==1).cpu()
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig)

    Y = torch.cat(Y, dim=0).numpy()
    Y = np.array(Y)
    Y_hat = torch.cat(Y_hat, dim=0).numpy() 
    Y_hat = np.array(Y_hat)    
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]
    
    return y_true, y_pred
    
if __name__=="__main__":

    labels = ['B-BODY',
      'B-DISEASES',
      'B-DRUG',
      'B-EXAMINATIONS',
      'B-TEST',
      'B-TREATMENT',
      'I-BODY',
      'I-DISEASES',
      'I-DRUG',
      'I-EXAMINATIONS',
      'I-TEST',
      'I-TREATMENT',
      'E-BODY',
      'E-DISEASES',
      'E-DRUG',
      'E-EXAMINATIONS',
      'E-TEST',
      'E-TREATMENT',
      'S-BODY',
      'S-DISEASES',
      'S-DRUG',
      'S-EXAMINATIONS',
      'S-TEST',
      'S-TREATMENT']
    
    best_model = None
    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--trainset", type=str, default="./CCKS_2019_Task1/processed_data/train_dataset.txt")
    parser.add_argument("--validset", type=str, default="./CCKS_2019_Task1/processed_data/val_dataset.txt")
    parser.add_argument("--testset", type=str, default="./CCKS_2019_Task1/processed_data/test_dataset.txt")
    # 将修改，换成自己的dataset
    
    ner = parser.parse_args(args=[])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    # 0516 这里改啦
    # 8.19
    max_len = 10000
    #model = BiLSTM_CRF(ner.batch_size, max_len, tag2idx).cuda()
    #model = BertME(tag2idx).cuda()
    model = Bert_BiLSTM_CRF(tag2idx).cuda()

    print('Initial model Done.')
    train_dataset = NerDataset(ner.trainset)
    eval_dataset = NerDataset(ner.validset)
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=PadBatch,
                                 drop_last=True)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=(ner.batch_size),
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch,
                                 drop_last=True)
    
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size),
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch,
                                drop_last=True)
    '''
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=(ner.batch_size)//2,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=PadBatch,
                                 drop_last=True)
    '''
    '''
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch,
                                drop_last=True)
    '''
    #optimizer = optim.Adam(self.model.parameters(), lr=ner.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset) 
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epoch
    
    warm_up_ratio = 0.1 # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    print("----This is BERT - BiLSTM - CRF----")
    print("----==== batch = 16 shuffle = False Learning rate = 0.001 两层LSTM ====----")
    print('Start Train...,')
    
    start = datetime.datetime.now()
    for epoch in range(1, ner.n_epochs+1):
        
        train(epoch, model, train_iter, optimizer, scheduler, device)
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device)
        ## 验证！
        ## model指的是bert-bilstm-crf
        ## epoch就是迭代次数
        ## eval_iter是DataLoader的结果

        if loss < _best_val_loss and acc > _best_val_acc:
            best_model = candidate_model
            _best_val_loss = loss
            _best_val_acc = acc
        
        print("=============================================")
    end = datetime.datetime.now()
    
    model_save_dir = "./modelsLiu"
    model_name = 'model1.pt'
    model_save_path = os.path.join(model_save_dir, model_name)

    best_model_state = copy.deepcopy(best_model.state_dict()) 
    torch.save(best_model_state, model_save_path)
    
    y_test, y_pred = test(best_model, test_iter, device)
    # print(y_test)
    # y_test表示测试集里面真实标签
    # y_pred表示测试集里的预测值
    Newy_test = [i.replace('B-', '').replace('I-', '').replace('E-', '').replace('S-', '') for i in y_test]
    Newy_pred = [i.replace('B-', '').replace('I-', '').replace('E-', '').replace('S-', '') for i in y_pred]
    Newlabels = ['BODY','DISEASES','DRUG','EXAMINATIONS','TEST','TREATMENT']
    print(metrics.classification_report(Newy_test, Newy_pred, labels=Newlabels, digits=3))
    print(end - start)