import argparse
import torch
import torch.nn as nn
from network.Math_Module import P, Q
from network.decom import Decom
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
from utils import *
import cv2

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)


class Inference(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # loading decomposition model 
        self.model_Decom_low = Decom()
        device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts.Decom_model_low_path)
        # loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding(self.opts.unfolding_model_path)
        # loading adjustment model
        self.adjust_model = load_adjustment(self.opts.adjust_model_path)

        # 选择使用 GPU 或 CPU

        self.device = device  # 保存 device 变量供后续使用
        self.model_Decom_low = self.model_Decom_low.to(device)  # 将模型移动到 GPU
        self.model_R = self.model_R.to(device)
        self.model_L = self.model_L.to(device)
        # self.unfolding_opts = self.unfolding_opts.to(device)
        self.adjust_model = self.adjust_model.to(device)  # 将模型移动到 GPU


        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        print(self.model_Decom_low)
        print(self.model_R)
        print(self.model_L)
        print(self.adjust_model)
        # time.sleep(8)

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):
            if t == 0:  # initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else:  # update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L

    def lllumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * self.opts.ratio
        return self.adjust_model(l=L, alpha=ratio)

    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start = time.time()
            R, L = self.unfolding(input_low_img)
            High_L = self.lllumination_adjust(L, self.opts.ratio)
            I_enhance = High_L * R
            p_time = (time.time() - start)
        return I_enhance, p_time

    def run(self, low_img_path):
        file_name = os.path.basename(self.opts.img_path)
        name = file_name.split('.')[0]
        low_img = self.transform(Image.open(low_img_path)).unsqueeze(0)
        enhance, p_time = self.forward(input_low_img=low_img)
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output,
                                 file_name.replace(name, "%s_%d_URetinexNet" % (name, self.opts.ratio)))
        np_save_TensorImg(enhance, save_path)
        print("================================= time for %s: %f============================" % (file_name, p_time))

    # # gradio 获得图片之后，传给训练函数的是 numpy类型数据
    # # 传入图片数据
    def runForWeb(self, image):
        # 对于配置低的服务器可以先对图片下采样训练，再上采样返回
        # 首先对输入的图片进行下采样直到符合最低运行像素限制
        max_pixel_limit = 600 * 600
        pyr_down_times = 0
        while True:
            a = len(image)
            b = len(image[0])
            c = a * b
            if (c <= max_pixel_limit):
                break
            pyr_down_times += 1
            image = cv2.pyrDown(image)

            # 对numpy数据预处理
        low_img = self.transform(Image.fromarray(np.uint8(image))).unsqueeze(0)

        # 开始训练
        enhance, p_time = self.forward(input_low_img=low_img)

        # 这里需要修改一下 utils.py 的结果返回给函数，
        # 参考原先的 run 函数 np_save_TensorImg 这里需要修改一下的位置
        # 退训练结果进行上采样，还原原图大小
        result_image = result_for_gradio(enhance)
        for i in range(pyr_down_times):
            result_image = cv2.pyrUp(result_image)

            # 返回 numpy 类型给 gradio 接口
        return result_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    # specify your data path here!
    parser.add_argument('--img_path', type=str, default="./demo/input/3.png")
    parser.add_argument('--output', type=str, default="./demo/output")
    # ratio are recommended to be 3-5, bigger ratio will lead to over-exposure 
    parser.add_argument('--ratio', type=int, default=5)
    # model path
    parser.add_argument('--Decom_model_low_path', type=str, default="./ckpt/init_low.pth")
    parser.add_argument('--unfolding_model_path', type=str, default="./ckpt/unfolding.pth")
    parser.add_argument('--adjust_model_path', type=str, default="./ckpt/L_adjust.pth")
    parser.add_argument('--gpu_id', type=int, default=0)

    opts = parser.parse_args()
    for k, v in vars(opts).items():
        print(k, v)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    model = Inference(opts).cuda()
    model.run(opts.img_path)
