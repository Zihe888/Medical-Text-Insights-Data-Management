# Medical-Text-Insights-Data-Management
## basic information about the pipeline files
- emr_pipeline.py is the pipeline runs on ec2 
- process_data.py is the pipelien runs on the local machine
- the workflows in the above files are orchestrated using Airflow 
## basic information about the deep learning models
- thismain.py is used to train the model
- thismodels.py is the model structure
- crf.py is the helper to build the model structure
- model.pt saves the parameters of the trained model
- processedTESTdataset.py is used to process the dataset
## setting up the cloud environment
- you need to install airflow and mongodb on EC2 instance
- use AWS Lambda and Amazon EventBridge to automatically stop and start Amazon EC2 instances
- make sure your EC2 instance can access your S3 bucket
- you can allocate memory on an EC2 instance to work as swap space to cope with the situation where you just need a little bit more memory to perform a task
