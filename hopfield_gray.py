import numpy as np
import random
from PIL import Image, ImageFilter
import os
import re
import time
import torch
from data_process import numpy_to_pic

def mat2vec(x):
    vec = x.reshape(-1)
    return vec


def create_W(x):
    w = torch.from_numpy(x)
    w = w.outer(w)
    return w
    


def output_array_to_pic(data, output_folder = None, i = 0):
    """
    将{0,1}值的numpy数组转化为灰度图片

    param:
        data:二维0-1numpy数组
        output_folder:图片保存路径,若为空则不保存图片
        i:图片索引
    output:
        img:灰度图片(黑白)
    """
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data == 1] = 255
    y[data == -1] = 0
    img = Image.fromarray(y)
    if output_folder != None:
        img.save(os.path.join(output_folder, f"image_{i + 1}.png"))
    return img

def array_to_pic(data, output_folder = None, i = 0):
    """
    将numpy数组转化为图片

    param:
        data:二维numpy数组(e.g.[92x92])
        output_folder:图片保存路径,若为空则不保存图片
        i:图片索引
    output:
        img:灰度图片
    """
    data = (data * 255).astype(np.int16)
    img = Image.fromarray(data)
    if output_folder != None:
        img.save(os.path.join(output_folder, f"image_{i + 1}.png"))
    return img


def update(w,y_vec,theta=0.5):
    y = torch.from_numpy(y_vec)
    y = y.reshape(-1,1)
    res = np.array(torch.mm(w,y)).reshape(-1)-theta
    res[res>=0]=1
    res[res<0]=-1
    return res 

def array_to_standard(array,threshold=145):
    """
    图像像素复原并且将图像像素按照阈值设置为{0,1}
    param:
        array:二维numpy数组
        threshold:转化阈值(像素值大于145则设为1,反之-1)
    output:
        x:转化后的numpy矩阵
    """
    img = (array * 255).astype(np.int16)
    x = np.zeros(img.shape,dtype=np.int16)
    x[img > threshold] = 1
    x[x==0] = -1
    return x 

def flat_numpy(array):
    """
    去除标准numpy图像数组的最后一个维度 例:[92x92x1] to [92x92]

    param:
        array:三维numpy数组
    output:
        *:二维numpy数组
    """
    return np.squeeze(array, axis=-1)



def train_test(train_path, test_path, res_path, theta=0.5, threshold=60):
    print("Importing images and creating weight matrix....")
    print(train_path)
    x = np.load(train_path)
    y = np.load(test_path)
    w = []
    count =0 
    for pic,noise_pic in zip(x,y):
        pic = flat_numpy(pic)
        noise_pic = flat_numpy(noise_pic)
        noise_pic_shape = noise_pic.shape
        # array_to_pic(pic).show()
        # array_to_pic(noise_pic).show()
        pic = array_to_standard(pic, threshold=threshold)
        noise_pic =array_to_standard(noise_pic, threshold=threshold)
        pic_vec = mat2vec(pic)
        noise_pic = mat2vec(noise_pic)
        print(len(pic_vec))
        start_time = time.time()
        w = create_W(pic_vec)
        end_time = time.time()
        print("Weight matrix is done!! Time costed:%fs"%(end_time-start_time))
        print("Imported test data")
        print("updating")
        y_vec_after = update(w=w,y_vec=noise_pic,theta=theta)
        y_vec_after = y_vec_after.reshape(noise_pic_shape)
        img = output_array_to_pic(y_vec_after,res_path,count)
        end_time = time.time()
        print("Update is done! Time costed:%fs"%(end_time-start_time))
        count = count + 1
        #img.show()

train_paths = './train_pics/processed_numpy_dataset/ORL.npy'
test_paths = './test_pics/processed_numpy_test/ORL_random.npy'
res_paths = './res_pics/ORL/random_block'

if __name__ == "__main__":
    train_test(train_path=train_paths, test_path=test_paths, res_path=res_paths, theta=0.5, threshold=138)