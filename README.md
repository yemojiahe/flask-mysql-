# Web

该项目配合一些健康检测设备使用，属于简单的开发项目，适合初入门学习 Flask 框架的同学。

## 功能特点
1. 前后端数据传输
2. 前端图表渲染
3. 数据库连接调用
4. WebSocket 使用
5. 上传数据功能以及界面美化
6. 简单调用 API 方法

## 技术栈
- 后端选用 Flask + SQLAlchemy + Flask-SocketIO  
- 前端使用 HTML + Layui + ECharts.js

## 项目结构

- `text.py` 为启动文件，`main.py` 为检测模块，登录密码为 1，账号为 1。
- 由于项目代码仅在一个模块中编写，项目主要聚焦于基本功能的实现，没有额外的安全控制。如果您愿意修改重构，可以上传我的仓库进行分享学习。

##使用方法

- 运行`text.py` ，在浏览器输入[loaclhost:5000](http://127.0.0.1:5000/login)打开网址
---

# Web

This project is designed to be used with some health monitoring devices and is a simple development project suitable for beginners learning the Flask framework.

## Features
1. Data transmission between front-end and back-end
2. Front-end chart rendering
3. Database connection and usage
4. WebSocket implementation
5. Data upload functionality and interface beautification
6. Simple API method calls

## Tech Stack
- Backend: Flask + SQLAlchemy + Flask-SocketIO  
- Frontend: HTML + Layui + ECharts.js

## Project Structure

- `text.py` is the startup file, and `main.py` is the detection module. The login credentials are username: 1, password: 1.
- Since the project code is written in a single module, it focuses mainly on the implementation of basic features and does not include additional security controls. If you wish to modify or restructure it, feel free to upload it to my repository for sharing and learning.
