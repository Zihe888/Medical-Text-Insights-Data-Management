## process multiple emr files into one json file (the name of the emr file is the patient id)
## patient id and the content will be stored in MongoDB (futrue step in another file)

## modules ##
import datetime
from collections import defaultdict
import os
import glob
import json
from datetime import date
from datetime import datetime
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

## data integration
def process_data(read_path: str, save_path: str) -> None:
    # path: directory to store the data
    # the name of the foler is the date
    
    data_list = []
    patient_list = {}
    count = -1
    # Use the glob module to match all txt files in the folder
    txt_files = glob.glob(os.path.join(read_path, '/[0-9]_patient_emr.txt'))

    # Iterate through each txt file and read its content
    for txt_file in txt_files:
        data_dict = defaultdict()
        with open(txt_file, 'r') as file:
            # emr data
            content = file.read()
            # patient id
            patient_id = txt_file.split('_')[0]
        if patient_id in patient_list.keys():
            index = patient_list[patient_id]
            data_temp = data_list[index]
            data_temp['emrs'].append(content)
        else:
            count += 1
            patient_list[patient_id] = count
            data_dict['patient_id'] = patient_id
            data_dict['emrs'] = list()
            data_dict['emrs'].append(content)
            data_list.append(data_dict)

    jsonfile = json.dumps(data_list, indent=2)
    # save data
    with open(save_path, 'w') as json_file:
        json_file.write(jsonfile)

## backup the data on amazon S3
def backup_data(filename: str, key: str, bucket_name: str) -> None:
    hook = S3Hook('s3_conn')
    hook.load_file(filename=filename, key=key, bucket_name=bucket_name)

## delete local file
def delete_data(path: str, name: str) -> None:

    file_path = path + name
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {file_path} : {e.strerror}")
    
    remind_file = path + 'remind.txt'
    with open(remind_file, 'w') as txt_file:
        txt_file.write("have backed up files on Amazon S3")
     

current_date = date.today()
formatted_date = current_date.strftime("%Y%m%d")
read_path = 'D://raw_data//' + formatted_date + '//'
save_path = 'D://upload_data//'
name = formatted_date + '.json'
filename = save_path + name

with DAG(
    dag_id='process_date',
    schedule_interval='@daily',
    start_date=datetime(2023, 6, 2, 18, 0),
    catchup=False
) as dag:
    
    task_to_process = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        op_kwargs={
            'read_path': read_path, 
            'save_path': filename
        }
    )

    # Upload the file
    task_upload_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=backup_data,
        op_kwargs={
            'filename': filename,
            'key': name,
            'bucket_name': 'emrs-backup-bucket'
        }
    )

    task_delete_data = PythonOperator(
        task_id='delete_data',
        python_callable=delete_data,
        op_kwargs={
            'path': save_path,
            'name': name
        }
    )

task_to_process >> task_upload_to_s3 >> task_delete_data