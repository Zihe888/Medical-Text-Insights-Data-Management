# -*- coding: utf-8 -*-

from flask import Flask    # import flask
from flask import render_template
from flask import request
import thispredict as pre

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
        #FileName = input("请输入电子病历名字,不用加后缀")
        ret = pre.predict(FileName,Model_BertBilstmCrf,device)
    
    return ret

if __name__ == "__main__":

    device = 'cpu'
    model_save_dir = "./"
    model_name = 'model.pt'
    Model_BertBilstmCrf = pre.LoadModel(model_save_dir,model_name,device) 
    
    app.run(port=2020,host="127.0.0.1",debug=True)   #调用run方法，设定端口号，启动服务
