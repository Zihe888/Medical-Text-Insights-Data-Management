from datetime import datetime, timedelta

from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow import DAG
from airflow.providers.mongo.hooks.mongo import MongoHook

# -*- coding: utf-8 -*-
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 导入pymysql模块
import torch
from torch.utils import data
import warnings
import argparse

from thismodels import Bert_BiLSTM_CRF
from utils1 import NerDataset, PadBatch, tag2idx, idx2tag
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

    y_pred = []
    for y_sentence in Y_hat:
        temp = []
        for i in y_sentence:
            if(int(i) != 1 and int(i) != 2):
                temp.append(idx2tag[int(i)])

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

##### if airflow cannot pass 'model' parameter, then integrate this function with predict function
def LoadModel(model_save_dir,model_name,device): 
    
    print("Load the model...")
    model = Bert_BiLSTM_CRF(tag2idx).cpu()
    model.to(device)

    model_save_path = os.path.join(model_save_dir, model_name)
    if os.path.exists(model_save_path):        
        loaded_paras = torch.load(model_save_path,map_location=device)        

    model.load_state_dict(loaded_paras,strict=False)  # Reinitialize network weight parameters with locally available models
    
    return model

def predict(FileName, device, model_save_dir, model_name):

    Model_BertBilstmCrf = LoadModel(model_save_dir,model_name,device)
    test_iter = loadTestData(FileName)
    
    print("----This is BERT - BiLSTM - CRF Model----")
    print("predict the result")
    y_pred = test(Model_BertBilstmCrf, test_iter, device)
    return y_pred

def ProcessRes(f_path, **kwargs):
    
    predict_res = kwargs['task_instance'].xcom_pull(task_ids="make_prediction_task")
    originaldata = Originaldata(f_path)
    res_sentence = []
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
                ## From left to right are sentence number, word type, position start point, position end point, original words
                temp.append(word_res)

        res_sentence.append(temp) 

    return res_sentence  

def ChangeOriType(f_path):

    test_dataset = Originaldata(f_path)
    temp = []
    for sentence in test_dataset:
        sent = "".join(sentence)
        temp.append(sent)
    
    final = "".join(temp)
    return final

def Originaldata(f_path):
    MAX_LEN = 256 - 2
    sents = []

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


def SaveDatatoDB(patientid, **kwargs):

    id = int(patientid)
    ori_data = kwargs['ti'].xcom_pull(task_ids="proccess_rori_task")
    res = kwargs['ti'].xcom_pull(task_ids="proccess_res_task")

    ## create mongodbhook ##
    with MongoHook(conn_id='airflow_mongo') as mongo_hook:
        
        collection_name = "EMRDataset"

        EMR = {'original': ori_data, 'predicted': res}
        filter_criteria = {'patientid': id}
        update_operation = {'$addToSet': {'EMRs': EMR}}
            
        try:
            ## modify part of the document
            result = mongo_hook.update_one(
                mongo_collection = collection_name,
                filter_doc=filter_criteria, 
                update_doc=update_operation,
                mongo_db='airbnb', 
                upsert=True)       
            
            if result.modified_count > 0 or result.upserted_id is not None:
                print("Update successful")
        except Exception as e:
            print(f"Error updating document with id {id}: {e}")          


if __name__ == "__main__":
    ## read files & make prediction
    ## filepath
    filepath = "D:/Liuuu/Undergraduate/GraduationProject/Codes/Bert-Bilstm-CRF/ChineseMedicalEntityRecognitionmaster/newfile"
    ### load model
    device = 'cpu'
    model_save_dir = filepath
    model_name = 'model.pt'
    ## Model_BertBilstmCrf = pre.LoadModel(model_save_dir,model_name,device) 

    ### make prediction
    FileName = 'test0'
    PatientID = '4'
    path = filepath + "/CCKS_2019_Task1/processed_data/"+FileName+".txt"
    ## execute query

    default_args = {
        "owner": "airflow",
        "start_date": datetime(2024, 1, 15),
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
        'timezone': 'PST'
    }

    with DAG(
        "pipeline",
        default_args=default_args,
        description="conduct ETL process of Medical EMRs data",
        schedule=timedelta(days=1), ## run this dag everyday
    ) as dag_EMR:
        
        start_task = EmptyOperator(task_id="ETL", dag=dag_EMR)

        make_prediction_task = PythonOperator(
            task_id="make_prediction_task",
            python_callable=predict,
            op_kwargs={'FileName': FileName,'device': device, 'model_save_dir': model_save_dir, 'model_name': model_name},
            provide_context=True,
            dag=dag_EMR,
        )

        proccess_res_task = PythonOperator(
            task_id="proccess_res_task",
            python_callable=ProcessRes,
            op_kwargs={'f_path': path},
            provide_context=True,
            #do_xcom_push=True,
            dag=dag_EMR,
        )  

        proccess_ori_task = PythonOperator(
            task_id="proccess_ori_task",
            python_callable=ChangeOriType,
            op_kwargs={'f_path': path},
            provide_context=True,
            #do_xcom_push=True,
            dag=dag_EMR,
        ) 

        # mongodata_name_review = 'hotel_review_text'
        load_emrs_mongo_task = PythonOperator(
            task_id="load_emrs_mongo_task",
            python_callable=SaveDatatoDB,
            #provide_context=True,
            #do_xcom_push=True,
            dag=dag_EMR,       
        )

    start_task >> make_prediction_task >> [proccess_res_task, proccess_ori_task] >> load_emrs_mongo_task
