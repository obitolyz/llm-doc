# FastChat 架构
fastchat由四个组件组成，分别为
- gradio web server 用于简单的 UI 交互
- Controller 用于管理 `model worker` 的模型名和模型地址
- openai_api server 提供与 **openai** 兼容的 API
- model worker 运行大模型，向 Controller 发送心跳

![fastchat_framework](images/fastchat_framework.png)

组件的启动顺序是先启动 `Controller` 模块，用于 model-worker 的注册信息。其次是 `model worker`，运行大模型，最后是 `gradio web server` 和 `openai_api server` 。