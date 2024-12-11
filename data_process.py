import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np


ORL_dataset_path = './datasets/ORL'  
kodak24_dataset_path = './datasets/Kodak24'


def load_orl_dataset(dataset_path, image_size=(100, 100)):
    """
    加载 ORL 数据集并转换为 NumPy 数组
    :param dataset_path: ORL 数据集的根目录
    :param image_size: 图片大小 (默认将图片缩放为 92x92)
    :param train_split: 每个类别的前 train_split 张图片作为训练集
    :return: 训练集和测试集的 NumPy 数组 (X_train, y_train, X_test, y_test)
    """
    X_train = []

    # 遍历每个类别的子文件夹
    for label, person_dir in enumerate(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            images = sorted(os.listdir(person_path))  # 确保按顺序加载
            for i, img_name in enumerate(images):
                img_path = os.path.join(person_path, img_name)
                # 读取图片并调整大小
                img = Image.open(img_path).convert('L')  # 转为灰度图
                img_resized = img.resize(image_size)
                img_array = np.asarray(img_resized, dtype=np.float32) / 255.0  # 归一化
                X_train.append(img_array)


    # 转换为 NumPy 数组
    X_train = np.array(X_train).reshape(-1, image_size[0], image_size[1], 1)  # 添加通道维度

    return X_train

def load_kodak24_dataset(dataset_path, image_size=(64, 64,3)):
    """
    加载 ORL 数据集并转换为 NumPy 数组
    :param dataset_path: ORL 数据集的根目录
    :param image_size: 图片大小 (默认将图片缩放为 92x92)
    :param train_split: 每个类别的前 train_split 张图片作为训练集
    :return: 训练集和测试集的 NumPy 数组 (X_train, y_train, X_test, y_test)
    """
    X_train = []

    images = sorted(os.listdir(dataset_path))  # 确保按顺序加载
    for i, img_name in enumerate(images):
        img_path = os.path.join(dataset_path, img_name)
        # 读取图片并调整大小
        img = Image.open(img_path)  # 转为灰度图
        img_resized = img.resize(image_size)
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0  # 归一化
        X_train.append(img_array)


    # 转换为 NumPy 数组
    X_train = np.array(X_train) 

    return X_train

def numpy_to_pic(array,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    # 转换为图片对象
    for i in range(array.shape[0]):
        # 提取图片数据并去掉最后一个通道维度(灰度图像)
        if(array.shape[-1]==1):
            img_data = np.squeeze(array, axis=-1)[i]
        else:
            img_data = array[i]
        # 转换为 0-255 范围的 uint8 类型（假设输入数据范围为 0-1）
        img_data = (img_data * 255).astype(np.uint8)

        # 转换为 PIL 图像
        img = Image.fromarray(img_data)

        # 保存图片
        img.save(os.path.join(output_folder, f"image_{i + 1}.png"))

    print(f"Saved {array.shape[0]} images to {output_folder}")
    return img


if __name__ == "__main__":
    # 数据预处理
    ORL_train_images = load_orl_dataset(ORL_dataset_path, image_size=(100, 100))
    kodak24_train_images = load_kodak24_dataset(kodak24_dataset_path, image_size=(100,100))
    ORL_numpy = ORL_train_images
    kodak24_numpy = kodak24_train_images

    selected_indices_ORL = np.random.choice(ORL_numpy.shape[0], size=10, replace=False)
    selected_indices_kodak24 = np.random.choice(kodak24_numpy.shape[0], size=10, replace=False)

    ORL_numpy = ORL_numpy[selected_indices_ORL]
    kodak24_numpy = kodak24_numpy[selected_indices_kodak24]

    np.save('./train_pics/processed_numpy_dataset/ORL.npy',ORL_numpy)
    np.save('./train_pics/processed_numpy_dataset/kodak24.npy',kodak24_numpy)

    # numpy数组转化为图片
    numpy_to_pic(ORL_numpy,'./train_pics/ORL/')
    numpy_to_pic(kodak24_numpy,'./train_pics/kodak24/')

