# from flask import Flask, render_template, jsonify
# from flask_socketio import SocketIO, emit
# import datetime
# import random
# import threading
# import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# 用于存储实时数据的全局变量
data = []
value_index = 0

# 模拟生成实时数据的函数
def generate_data():
    global data, value_index
    while True:
        now = datetime.datetime.now()
        value = random.randint(0, 100)  # 模拟一个随机值
        new_data = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "value": value
        }
        data.append(new_data)
        if len(data) > 100:
            data = data[-100:]  # 保留最后100条数据
        socketio.emit('data_update', new_data, namespace='/data')
        time.sleep(1)  # 每隔1秒生成一次数据

# 开启一个线程生成实时数据
thread = threading.Thread(target=generate_data)
thread.start()

# 定义路由，渲染包含 ECharts 图表的页面
@app.route('/')
def index():
    return render_template('1.html')

# WebSocket 的事件处理函数，用于接收前端的连接
@socketio.on('connect', namespace='/data')
def handle_connect():
    print('Client connected')

# 主程序入口
if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)

