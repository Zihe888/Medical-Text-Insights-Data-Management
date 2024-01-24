# -*- coding: utf-8 -*-
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 导入pymysql模块
import pymysql
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
#import sys   #导入sys模块
#sys.path.append("D:\\Liuuu\Undergraduate\\GraduationProject\\Codes\\Bert-Bilstm-CRF\\ChineseMedicalEntityRecognitionmaster")
#from ChineseMedicalEntityRecognitionmaster import utils
from utils1 import NerDataset, PadBatch, VOCAB, tokenizer, tag2idx, idx2tag
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
        
def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #print(batch)
            x, z = batch
            #print(x)
            y_hat = model(x, z)
            
            # Save prediction
            y_hat = torch.squeeze(y_hat)
            mask = (z==1)
            y_hat_ = torch.masked_select(y_hat, mask)
            Y_hat.append(y_hat_.cpu())

    #print(Y_hat)
    #print(len(Y_hat))
    #Y_hat = np.array(Y_hat)    
    y_pred = []
    for y_sentence in Y_hat:
        temp = []
        for i in y_sentence:
            #if((int(i) != 1) or (int(i) != 2)):
            if(int(i) != 1 and int(i) != 2):
                temp.append(idx2tag[int(i)])
                #print(temp)
        y_pred.append(temp)
    
    return y_pred

def loadTestData(FileName):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--testset", type=str, default="D:/Liuuu/Undergraduate/GraduationProject/Codes/BertBilstmCRF/ChineseMedicalEntityRecognitionmaster/CCKS_2019_Task1/processed_data/"+FileName+".txt")
    
    ner = parser.parse_args(args=[])

    print('Initial model Done.')
    test_dataset = NerDataset(ner.testset)
    #print(test_dataset)
    print('Load Data Done.')
    
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size),
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch,
                                drop_last=True)
    
    return test_iter

def LoadModel(model_save_dir,model_name,device):
    
    print("Load the model...")
    model = Bert_BiLSTM_CRF(tag2idx).cpu()
    model.to(device)

    model_save_path = os.path.join(model_save_dir, model_name)
    if os.path.exists(model_save_path):        
        loaded_paras = torch.load(model_save_path,map_location=device)        

    model.load_state_dict(loaded_paras,strict=False)  # 用本地已有模型来重新初始化网络权重参数
    
    return model

def predict(FileName,Model_BertBilstmCrf,device):

    test_iter = loadTestData(FileName)
    
    print("----This is BERT - BiLSTM - CRF----")
    print("----==== batch = 16 shuffle = False Learning rate = 0.001 两层LSTM ====----")
 
    
    print("predict the result")
    start = datetime.datetime.now()
    y_pred = test(Model_BertBilstmCrf, test_iter, device)
    #y_test, y_pred = test(model, test_iter, device)
    end = datetime.datetime.now()
    #print(end - start)
    #print(y_pred)
    #print(metrics.classification_report(Newy_test, Newy_pred, labels=Newlabels, digits=3))
    return y_pred

def ProcessRes(predict_res,originaldata):
    res_sentence = []
    count = 0
    start = 0
    end = 0
    for j in range(len(predict_res)):
        temp = []
        for i in range(len(predict_res[j])): 
            
            if predict_res[j][i][0] == 'O':
                continue 
            
            if predict_res[j][i][0] == 'B':
                start = i + 1
            elif predict_res[j][i][0] == 'E':
                end = i + 1
                ori = originaldata[j][(start - 1):end]
                finaldata = "".join(ori)
                length = end - start + 1
                word_res = [j + 1, predict_res[j][i][2:], finaldata, start, length]
                temp.append(word_res)
            elif predict_res[j][i][0] == 'S':
                start = i + 1
                end = i + 1
                ori = originaldata[j][(start - 1):end]
                finaldata = "".join(ori)
                length = end - start + 1
                word_res = [j + 1, predict_res[j][i][2:], finaldata, start, length]
                ## 从左边到右边分别是句子编号，词语类型，位置起点，位置终点，原文词语
                temp.append(word_res)

        res_sentence.append(temp) 

    return res_sentence  

def Originaldata(f_path):
    MAX_LEN = 256 - 2
    sents = []
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
                sents.append(word[:MAX_LEN])
            else:
                sents.append(word)
            word = []

    return sents

def SaveDatatoDB(res, ori_data, UserId, FileName):

    # 连接database
    conn = pymysql.connect(
        host='localhost', 
        port=3306,
        user='root',
        password='root123',
        database='intern',
        charset="utf8"
    )

    # 得到一个可以执行SQL语句的光标对象
    cursor = conn.cursor()
    # 定义要执行的SQL语句
    
    sql = "INSERT INTO emr_overview(userid, filename, fileContent) VALUES (%s, %s, %s);"
    userid = UserId
    filename = FileName
    fileContent = ori_data
    # 执行SQL语句
    cursor.execute(sql, [userid, filename, fileContent])
    # 执行SQL语句
    # cursor.execute(sql)
    
    for sentence in res:
        for word in sentence:
            sql = """INSERT INTO `emr_detail`(`userid`, `filename`, `sentenceID`, `type`, `start`, `length`) VALUES (%s, %s, %s, %s, %s, %s);"""
            #sql = """INSERT INTO emr_detail(userid, filename, sentenceID, type, start, length) VALUES (%s, %s, %s,%s, %s, %s);"""
            userid = int(UserId)
            filename = FileName
            sentenceID = str(word[0])
            type_name = word[1]
            #typeDetail = word[2]
            start = int(word[3])
            length = int(word[4])
            # 执行SQL语句
            cursor.execute(sql, [userid, filename, sentenceID, type_name, start, length])
            # 执行SQL语句
            #cursor.execute(sql)   
    
    # 关闭光标对象
    cursor.close()

    #提交保存
    conn.commit()

    # 关闭数据库连接
    conn.close()

def TestDatatoDB():
    # 连接database
    conn = pymysql.connect(
        host='localhost', 
        port=3306,
        user='root',
        password='root123',
        database='intern',
        charset="utf8"
    )

    # 得到一个可以执行SQL语句的光标对象
    cursor = conn.cursor()
    # 定义要执行的SQL语句
    sql = "INSERT INTO emr_overview(userid, filename, fileContent) VALUES (%s, %s, %s);"
    UserId = '12'
    FileName = '232'
    ori_data = '你好呀'
    userid = UserId
    filename = FileName
    fileContent = ori_data
    # 执行SQL语句
    cursor.execute(sql, [userid, filename, fileContent])
    # 执行SQL语句
    # cursor.execute(sql)
    #sql = """INSERT INTO `emr_detail`(`userid`, `filename`, `sentenceID`, `type`, `start`, `length`) VALUES (%s, %s, %s, %s, %s, %s);"""
    # 执行SQL语句
    #cursor.execute(sql, (1, 'filename', 'sentID', 'type_name', 0, 1))
    # 执行SQL语句
    #cursor.execute(sql)
    
    # 关闭光标对象
    cursor.close()

    conn.commit()
    # 关闭数据库连接
    conn.close()

def ChangeOriType(test_dataset):

    temp = []
    for sentence in test_dataset:
        sent = "".join(sentence)
        temp.append(sent)
    
    final = "".join(temp)
    return final

if __name__=="__main__":

    #TestDatatoDB()
    #exit()

    device = 'cpu'
    model_save_dir = "./"
    model_name = 'model.pt'
    Model_BertBilstmCrf = LoadModel(model_save_dir,model_name,device) 

    #FileName = input("请输入电子病历名字,不用加后缀")
    FileName = 'test'
    UserId = '4'
    path = "D:/Liuuu/Undergraduate/GraduationProject/Codes/BertBilstmCRF/ChineseMedicalEntityRecognitionmaster/CCKS_2019_Task1/processed_data/"+FileName+".txt"
    test_dataset = Originaldata(path)
    L = predict(FileName,Model_BertBilstmCrf,device)
    res = ProcessRes(L, test_dataset)
    process_ori = ChangeOriType(test_dataset)
    SaveDatatoDB(res, process_ori, UserId, FileName)