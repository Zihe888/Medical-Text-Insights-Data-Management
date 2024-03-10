# -*- coding: utf-8 -*-
from datetime import datetime, timedelta, date
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow import DAG
from airflow.providers.mongo.hooks.mongo import MongoHook
import boto3
from itertools import accumulate
import os
import torch
from torch.utils import data
import warnings
import argparse

from thismodels import Bert_BiLSTM_CRF
from utils import NerDataset, PadBatch, tag2idx, idx2tag
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  

def store_s3_file(bucket_name, file_key):

    collection_name = "EMRDataset"
    file_content = read_s3_file(bucket_name, file_key)
    with MongoHook(conn_id='airflow_mongo') as mongo_hook:
        for document in file_content:
            mongo_hook.insert_one(
                collection=collection_name,
                document=document,
                mongo_db='emrs')

## save the processed file on ec2
def process_s3_file(bucket_name, file_key, filename):

    file_content = read_s3_file(bucket_name, file_key)
    patient_ids = list()
    emrs_len = list()
    with open(filename, 'w', encoding='utf-8') as f:
        for file in file_content:
            patient_contents = file['emrs']
            for content in patient_contents:
                patient_ids.append(file['patient_id'])
                emrs_len.append(content)
                for char in content:
                    f.write(char)
                    f.write('\n')
    
    return patient_ids, emrs_len

## make prediction and store the result into MongoDB
def predict_savepredict(FileName, device, model_save_dir, model_name, **kwargs):

    patient_ids, emrs_len = kwargs['ti'].xcom_pull(task_ids="task_process_s3_file")
    Model_BertBilstmCrf = LoadModel(model_save_dir,model_name,device)
    test_iter = loadTestData(FileName)
    
    print("----This is BERT - BiLSTM - CRF Model----")
    print("predict the result")
    y_pred = test(Model_BertBilstmCrf, test_iter, device)
    subsent_temp = [y_pred[start:start+length] for start, length in zip([0] + list(accumulate(emrs_len)), emrs_len)]
    subsent_predict = ProcessRes(FileName, subsent_temp)

    for i in range(len(patient_ids)):
        id = i
        res = subsent_predict[i]
        with MongoHook(conn_id='airflow_mongo') as mongo_hook: 

            collection_name = "EMRDataset"
            EMR = {'emrs_predicted': res}
            filter_criteria = {'patientid': id}
            update_operation = {'$addToSet': {'EMRs': EMR}}
                
            try:
                ## modify part of the document
                result = mongo_hook.update_one(
                    mongo_collection = collection_name,
                    filter_doc=filter_criteria, 
                    update_doc=update_operation,
                    mongo_db='emrs', 
                    upsert=True)       
                
                if result.modified_count > 0 or result.upserted_id is not None:
                    print("Update successful")
            except Exception as e:
                print(f"Error updating document with id {id}: {e}")         

def delete_ec2file(file_path):
    
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {file_path} : {e.strerror}")
    


## helper for s3 file
def read_s3_file(bucket_name, file_key):
    s3_client = boto3.client('s3')
    try:
    
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
    
    except Exception as e:
    
        print("Error reading file from S3:", e)

    return file_content

## helper for predict
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

## helper for predict
def LoadModel(model_save_dir,model_name,device): 
    
    print("Load the model...")
    model = Bert_BiLSTM_CRF(tag2idx).cpu()
    model.to(device)

    model_save_path = os.path.join(model_save_dir, model_name)
    if os.path.exists(model_save_path):        
        loaded_paras = torch.load(model_save_path,map_location=device)        

    model.load_state_dict(loaded_paras,strict=False)  # Reinitialize network weight parameters with locally available models
    
    return model

## helper for predict
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

## helper for predict (process the predict result)
def ProcessRes(f_path, predict_res)->None:
    ## f_path original data path
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

## helper for loading original file
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

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 6, 2, 19, 0),
    'retries': 1,
}


current_date = date.today()
formatted_date = current_date.strftime("%Y%m%d")
# key
name = formatted_date + '.json'
# bucket name
bucket_name = 'emrs-backup-bucket'
## s3 file on ec2
filename = "yourpath" + name
## model path
model_save_dir = 'yourpath'
## device
device = 'cpu'
## model name
model_name = 'model.pt'

with DAG(
    dag_id='emr_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    description='A DAG to conduct emrs and insights data management'
) as dag_EMR:

    task_store_s3_file = PythonOperator(
        task_id="task_store_s3_file",
        python_callable=store_s3_file,
        op_kwargs={'bucket_name': bucket_name, 'file_key': name},
        dag=dag_EMR,
    )

    task_process_s3_file = PythonOperator(
        task_id="task_process_s3_file",
        python_callable=process_s3_file,
        op_kwargs={'bucket_name': bucket_name, 'file_key': name, 'filename': filename},
        dag=dag_EMR,
    )

    task_predict_savepredict = PythonOperator(
        task_id="",
        python_callable=predict_savepredict,
        op_kwargs={'FileName': filename, 'device': name, 'model_save_dir': model_save_dir, 'model_name': model_name},
        provide_context=True,
        dag=dag_EMR,
    )

    task_delete_ec2file = PythonOperator(
        task_id="task_delete_ec2file",
        python_callable=delete_ec2file,
        op_kwargs={'file_path': filename},
        dag=dag_EMR,       
    )


[task_store_s3_file, task_process_s3_file] >> task_predict_savepredict >> task_delete_ec2file