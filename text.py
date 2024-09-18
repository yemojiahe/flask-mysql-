from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
import numpy as np
import random
import datetime
import threading
from bot import send_message_to_bot ,requests
import os
import re
import base64
import json
# from ana import result_ana,loadfile
import time
from flask_socketio import SocketIO, emit
from flask_cors import CORS


import main

from werkzeug.utils import secure_filename



app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
# global_filepath = main.mysql2()
# file_path = 'updatefile\\filtered_signal.npy'
global_filepath = "signal.npy"
def load_initial_values(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return None  # 如果文件不存在，返回 None 或者适当的错误信息
        # 加载初始值
    initial_values = np.load(file_path)
    return initial_values

initial_values=load_initial_values(global_filepath)
value_index = 0
# print("第一个：",initial_values[value_index])
# 初始化数据
data = []

@app.route('/')
def index1():
    return render_template('login.html')

# 第二个页面路由
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    print(username)
    # 这里可以添加简单的用户名和密码验证逻辑
    if username == '1' and password == '1':
        # 登录成功，可以重定向到另一个页面或者返回成功的 JSON 响应
        return redirect(url_for('index'))
    else:
        # 登录失败，可以返回错误信息到前端
        return render_template('login.html', message='用户名或密码错误')


@app.route('/index')
def index():
    # 这里可以是登录成功后跳转的页面的逻辑
    return render_template('index.html')



@app.route('/1.html')
def chart():
    return render_template('1.html')





@app.route('/bot.html')
def bot():
    return render_template('bot.html')


@app.route('/2.html')
def sdk():
    return render_template('2.html')



@app.route('/information.html')
def information():
    # 这里可以是登录成功后跳转的页面的逻辑
    return render_template('information.html')

@app.route('/home.html')
def home():
    # 这里可以是登录成功后跳转的页面的逻辑
    return render_template('home.html')




#文件上传的
@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        img_file = request.files.get('pic')
        file_name = img_file.filename
        # 文件名的安全转换
        filename = secure_filename(file_name)
        # 保存文件
        UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'updatefile')
        img_file.save(os.path.join(UPLOAD_PATH, filename))
        global global_filepath
        global_filepath=os.path.join(UPLOAD_PATH, filename)
        return render_template('1.html')



@app.route('/upload2/', methods=['POST'])
def upload2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    img_file = request.files['file']  # 这里的'file'要与前端相对应
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'updatefile')

        # 确保上传目录存在
        os.makedirs(UPLOAD_PATH, exist_ok=True)

        filename = img_file.filename  # 获取文件名
        # 完整的文件保存路径
        global global_filepath
        global_filepath = os.path.join(UPLOAD_PATH, filename)
        img_file.save(global_filepath)  # 保存文件
        result = main.result_ana(global_filepath)  # 调用你的分析函数

        print("File uploaded and processed successfully.")
        return jsonify(result=result), 200  # 返回结果
    except Exception as e:
        print(f"Error occurred: {e}")  # 打印错误信息
        return jsonify({'error': str(e)}), 500  # 返回错误信息

#上传文件页面的接口
@app.route('/result')
def result():
    filepath = global_filepath
    result = main.result_ana(filepath)
    # result = 1
    return jsonify(result=result)



#用户的接口
@app.route('/result2')
def result2():
    # #得先返回生成本地文件
    filepath =global_filepath
    result = main.result_ana(filepath)
    return jsonify(result=result)





########
# @socketio.on('connect')
# def handle_connect():
#     global now, value_index
#     print('Client connected')
#
#     # Load initial values
#     initial_values = load_initial_values(global_filepath)
#     global now, value_index
#
#     if initial_values is not None:
#         count = 0
#         a=0
#         current_time = datetime.datetime.now()
#         for value in initial_values:
#             now = datetime.datetime.now()
#             data={
#                 "name": count,
#                 "value": value
#             }
#             time.sleep(0.01)
#             emit('new_data', data)
#             count += 0.001
#             a+=1
#             if a >= 500:
#                 print("完",time.time())
#                 break
@socketio.on('connect')
def handle_connect():
    print('Client connected')

    # Load initial values (assuming this function exists)
    initial_values = load_initial_values(global_filepath)

    if initial_values is not None:
        thread = threading.Thread(target=send_data_to_client, args=(initial_values,))
        thread.start()

def send_data_to_client(initial_values):
    count = 0
    a = 0
    current_time = datetime.datetime.now()
    for value in initial_values:
        now = datetime.datetime.now()
        data = {
            "name": 0,
            "value": value
        }
        socketio.emit('new_data', data)  # Using socketio.emit instead of emit

        a += 1
        if a >= 1000:
            print("Complete", time.time())
            break





@app.route('/data')
def get_data():
    global now, value_index
    now=datetime.datetime.now()
    initial_values=load_initial_values(global_filepath)
    if initial_values is None:
        return "NULL"
    else:
        value_index = (value_index + 1) % len(initial_values)
        new_data = {
            "name": now.strftime("%Y/%m/%d %H:%M:%S"),
            "value": [now.strftime("%Y/%m/%d %H:%M:%S"), initial_values[value_index-1]]

        }
        print(new_data)
        return jsonify(new_data)


#数据库来的数据
@app.route('/data2')
def get_data2():
    global now, value_index
    now=datetime.datetime.now()
    print(global_filepath)
    file_path=global_filepath        #代码不同的是这块
    print(file_path)
    initial_values=load_initial_values(file_path)
    if initial_values is None:
        return "NULL"
    else:
        value_index = (value_index + 1) % len(initial_values)
        new_data = {
            "name": now.strftime("%Y/%m/%d %H:%M:%S"),
            "value": [now.strftime("%Y/%m/%d %H:%M:%S"), initial_values[value_index - 1]]

        }
        print(new_data)
        return jsonify(new_data)


# @app.route('/send_message', methods=['POST'])
# def handle_message():
#     s = time.time()
#     accumulated_msgs = []
#     data = request.get_json()
#     print(data)
#     bot_id = '7392882322007572495'
#     user_id = '123456789'
#     message = data.get('message')
#
#     if not bot_id or not user_id or not message:
#         return jsonify({'error': 'Missing required parameters (bot_id, user_id, message)'}), 400
#
#     try:
#         print("正在发往message：", message)
#
#         for response_data in send_message_to_bot(bot_id, user_id, message):
#
#            print(response_data)
#
#         mid = time.time()
#         print("传输时间", mid - s)
#         a = list(i)  # 将生成器对象的值放入列表中
#
#         return jsonify(msg)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


@app.route('/2.html')
def text():
    return render_template('2.html')


@app.route('/send_msg', methods=['POST'])
def handle_msg():
    s=time.time()
    accumulated_msgs = []
    data = request.get_json()
    print(data)
    bot_id = '7392882322007572495'
    user_id = '123456789'
    message = data.get('message')


    if not bot_id or not user_id or not message:
        return jsonify({'error': 'Missing required parameters (bot_id, user_id, message)'}), 400

    try:
        print("正在发往message：",message)

        i = send_message_to_bot(bot_id, user_id, message)  # 获取生成器对象
        mid=time.time()
        print("传输时间",mid-s)
        a = list(i)  # 将生成器对象的值放入列表中
        print(a[-13])
        msg=a[-13]

        #json化发送
        prefix = 'data:'
        prefix_length = len(prefix)
        if msg.startswith(prefix):
            msg = msg[prefix_length:]
        print(msg)

        e=time.time()
        print("耗时：" ,e-s)
        return jsonify(msg)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


API_KEY = "sk-081cb15c9d884e59a456b7b07a4f9901qipi2rstq3nls9to"


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Encode the image to base64
    base64_image = base64.b64encode(file.read()).decode('utf-8')

    # Prepare payload for AI service
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "qwen-vl-8k",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "分析图片"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # Send request to AI service
    try:
        response = requests.post("https://ai-gateway.vei.volces.com/v1/chat/completions", headers=headers, json=payload)
        response= response.json()

        if "choices" in response and len(response["choices"]) > 0:
            msg = response["choices"][0]
            msg_str = json.dumps(msg, indent=2,
                                 ensure_ascii=False)
            a = msg_str.encode('utf-8').decode()

            print(a)
            return jsonify(a)
        else:
            return jsonify({'error': 'No description found'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True)

