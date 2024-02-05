# -*- coding: utf-8 -*-

from flask import Flask    #导入Flask类
from flask import render_template
from flask import request
from thispredict import *

app=Flask(__name__)         #实例化并命名为app实例

@app.route('/')
def index():
    #return 'welcome to my webpage!'
    msg="my name is caojianhua, China up!"
    return render_template("index.htm",data=msg)  #加入变量传递,index.html中用{{data}}接收信息

@app.route('/test',methods=['GET','POST'])
def testArg():
    if request.method=='POST':  #POST获取参数
        user = request.form['user']
        passwd = request.form['passwd']
        ret = user + "->" + passwd
    else:
        FileName = request.args.get("FileName")
        UserId = request.args.get("UserId")
        path = "/home/ubuntu/data/" + FileName + ".txt"
        ret = predict(FileName,Model_BertBilstmCrf,device)

        test_dataset = Originaldata(path)
        res = ProcessRes(ret, test_dataset)
        process_ori = ChangeOriType(test_dataset)
        SaveDatatoDB(res, process_ori, UserId, FileName)
    
    return 'done!'

if __name__ == "__main__":

    device = 'cpu'
    model_save_dir = "./"
    model_name = 'model.pt'
    Model_BertBilstmCrf = LoadModel(model_save_dir,model_name,device) 
    
    app.run(port=80,host="172.26.0.187",debug=True)   #调用run方法，设定端口号，启动服务

'''
# -*- coding: utf-8 -*-

from flask import Flask    #导入Flask类
from flask import render_template
from flask import request
import thispredict

app=Flask(__name__)         #实例化并命名为app实例

@app.route('/')
def index():
    #return 'welcome to my webpage!'
    msg="my name is caojianhua, China up!"
    return render_template("index.htm",data=msg)  #加入变量传递,index.html中用{{data}}接收信息

@app.route('/test',methods=['GET','POST'])
def testArg():
    if request.method=='POST':  #POST获取参数
        user = request.form['user']
        passwd = request.form['passwd']
        ret = user + "->" + passwd
    else:
        user = request.args.get("user")
        passwd = request.args.get("passwd")

        ret = "hellow " + user + "->" + passwd
    
    return ret

if __name__ == "__main__":
    app.run(port=2020,host="127.0.0.1",debug=True)   #调用run方法，设定端口号，启动服务
'''