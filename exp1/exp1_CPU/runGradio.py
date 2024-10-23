#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/14 13:37
# @Author  : 钟昊天2021280300
# @FileName: runGradio.py
# @Software: PyCharm

import gradio as gr
import test

# 手动创建 opts 对象
class Opts:
    def __init__(self):
        self.img_path = "demo/input/3.png"
        self.output = "./demo/output"
        self.ratio = 5
        self.Decom_model_low_path = "ckpt/init_low.pth"
        self.unfolding_model_path = "ckpt/unfolding.pth"
        self.adjust_model_path = "ckpt/L_adjust.pth"
        self.gpu_id = 0

opts = Opts()

# 创建 Gradio 接口，传入 test 模块中的 runForWeb 函数，输入类型为 image，输出类型为 image
model = test.Inference(opts)
interface = gr.Interface(fn=model.runForWeb, inputs='image', outputs='image')

# 启动 Gradio 接口，设置共享链接
interface.launch(share=True)

# 指定主机地址为本地访问，指定端口为 8888
interface.launch(server_name='127.0.0.1', server_port=8888)
