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
# 0819
from thismodels import Bert_BiLSTM_CRF
#from BERTmodels import BertME
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
        
def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            #x = x.to(device)
            #z = z.to(device)
            print(x)
            x = x.to(device)
            z = z.to(device)
            print(x)
            print("-----")
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
    parser.add_argument("--testset", type=str, default="D:/Liuuu/Undergraduate/GraduationProject/Codes/BertBilstmCRF/ChineseMedicalEntityRecognitionmaster/CCKS_2019_Task1/processed_data/test_dataset.txt")
    # 将修改，换成自己的dataset
    
    ner = parser.parse_args(args=[])
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    # 0516 这里改啦
    # 0819
    max_len = 10000
    #model = BiLSTM_CRF(ner.batch_size, max_len, tag2idx).cuda()
    #model = BertME(tag2idx).cuda()
    model = Bert_BiLSTM_CRF(tag2idx).cpu()

    print('Initial model Done.')
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')
    
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size),
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch,
                                drop_last=True)
    print(enumerate(test_iter))
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

    print("----This is BERT - BiLSTM - CRF----")
    print("----==== batch = 16 shuffle = False Learning rate = 0.001 两层LSTM ====----")
    
    print("Load the model...")
    ## 0819 loadmodel替换bestmodel    
    model.to(device)    
    
    model_save_dir = "./"
    model_name = 'model.pt'
    model_save_path = os.path.join(model_save_dir, model_name)
    if os.path.exists(model_save_path):        
        loaded_paras = torch.load(model_save_path,strict=False)        

    model.load_state_dict(loaded_paras)  # 用本地已有模型来重新初始化网络权重参数   
    
    print("predict the result")
    
    start = datetime.datetime.now()
    y_test, y_pred = test(model, test_iter, device)
    end = datetime.datetime.now()
    print(end - start)
    
    # print(y_test)
    # y_test表示测试集里面真实标签
    # y_pred表示测试集里的预测值
    Newy_test = [i.replace('B-', '').replace('I-', '').replace('E-', '').replace('S-', '') for i in y_test]
    Newy_pred = [i.replace('B-', '').replace('I-', '').replace('E-', '').replace('S-', '') for i in y_pred]
    Newlabels = ['BODY','DISEASES','DRUG','EXAMINATIONS','TEST','TREATMENT']
    print(metrics.classification_report(Newy_test, Newy_pred, labels=Newlabels, digits=3))