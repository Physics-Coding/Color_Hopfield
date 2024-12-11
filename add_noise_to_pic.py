"""
该部分对于彩色图像和单色图像均可以复用
"""


from data_process import numpy_to_pic
import numpy as np

ORL_gaussian_noise_path = './test_pics/ORL/gaussian'
ORL_poisson_noise_path = './test_pics/ORL/poisson'
ORL_random_block_noise_path = './test_pics/ORL/random_block'
ORL_salt_pepper_noise_path = './test_pics/ORL/salt_pepper'
ORL_uniform_noise_path = './test_pics/ORL/uniform_noise'

ORL_gausssian_numpy = './test_pics/processed_numpy_test/ORL_gaussian.npy'
ORL_poisson_numpy = './test_pics/processed_numpy_test/ORL_poisson.npy'
ORL_random_block_numpy = './test_pics/processed_numpy_test/ORL_random.npy'
ORL_salt_numpy = './test_pics/processed_numpy_test/ORL_salt.npy'
ORL_uniform_numpy = './test_pics/processed_numpy_test/ORL_uniform.npy'


kodak24_gaussian_noise_path = './test_pics/kodak24/gaussian'
kodak24_poisson_noise_path = './test_pics/kodak24/poisson'
kodak24_random_block_noise_path = './test_pics/kodak24/random_block'
kodak24_salt_pepper_noise_path = './test_pics/kodak24/salt_pepper'
kodak24_uniform_noise_path = './test_pics/kodak24/uniform_noise'

kodak24_gausssian_numpy = './test_pics/processed_numpy_test/kodak24_gaussian.npy'
kodak24_poisson_numpy = './test_pics/processed_numpy_test/kodak24_poisson.npy'
kodak24_random_block_numpy = './test_pics/processed_numpy_test/kodak24_random.npy'
kodak24_salt_numpy = './test_pics/processed_numpy_test/kodak24_salt.npy'
kodak24_uniform_numpy = './test_pics/processed_numpy_test/kodak24_uniform.npy'


def add_gaussian_noise(images, mean=0, std=0.1):
    """
    给图片添加高斯噪音
    :param images: 原始图片数组，形状为 (N, H, W, C) 或 (N, H, W)
    :param mean: 高斯噪音的均值
    :param std: 高斯噪音的标准差
    :return: 添加噪音后的图片数组
    """
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)  # 限制像素值范围在 [0, 1]
    return noisy_images

def add_salt_and_pepper_noise(images, prob=0.05):
    """
    给图片添加椒盐噪音
    :param images: 原始图片数组，形状为 (N, H, W, C) 或 (N, H, W)
    :param prob: 噪音比例，0 到 1 之间
    :return: 添加噪音后的图片数组
    """
    noisy_images = images.copy()
    for i in range(images.shape[0]):
        salt_pepper = np.random.rand(*images[i].shape)
        noisy_images[i][salt_pepper < prob / 2] = 0  # 椒（黑色）
        noisy_images[i][salt_pepper > 1 - prob / 2] = 1  # 盐（白色）
    return noisy_images


def add_poisson_noise(images):
    """
    给图片添加泊松噪音
    :param images: 原始图片数组，形状为 (N, H, W, C) 或 (N, H, W)
    :return: 添加噪音后的图片数组
    """
    noisy_images = np.random.poisson(images * 255.0) / 255.0  # 放大至 [0, 255]，再归一化
    noisy_images = np.clip(noisy_images, 0, 1)  # 限制像素值范围
    return noisy_images


def add_uniform_noise(images, low=-0.2, high=0.2):
    """
    给图片添加均匀噪音
    :param images: 原始图片数组，形状为 (N, H, W, C) 或 (N, H, W)
    :param low: 噪音范围下限
    :param high: 噪音范围上限
    :return: 添加噪音后的图片数组
    """
    noise = np.random.uniform(low, high, images.shape)
    noisy_images = images + noise
    noisy_images = np.clip(noisy_images, 0, 1)  # 限制像素值范围
    return noisy_images


def add_random_block_noise(images, block_size=20, num_blocks=5):
    """
    给图片添加随机遮挡噪音
    :param images: 原始图片数组，形状为 (N, H, W, C) 或 (N, H, W)
    :param block_size: 遮挡块的大小
    :param num_blocks: 每张图片添加的遮挡块数量
    :return: 添加噪音后的图片数组
    """
    noisy_images = images.copy()
    for i in range(images.shape[0]):
        h, w = images[i].shape[:2]
        for _ in range(num_blocks):
            x = np.random.randint(0, w - block_size)
            y = np.random.randint(0, h - block_size)
            noisy_images[i, y:y+block_size, x:x+block_size] = 0  # 遮挡块设为黑色
    return noisy_images

if __name__ == "__main__":
    ORL_array = np.load('./train_pics/processed_numpy_dataset/ORL.npy')
    kodak24_array = np.load('./train_pics/processed_numpy_dataset/kodak24.npy')

    # ORL
    ORL_array_gaussian = add_gaussian_noise(ORL_array)
    numpy_to_pic(ORL_array_gaussian,ORL_gaussian_noise_path)
    np.save(ORL_gausssian_numpy,ORL_array_gaussian)

    ORL_array_poisson = add_poisson_noise(ORL_array)
    numpy_to_pic(ORL_array_poisson,ORL_poisson_noise_path)
    np.save(ORL_poisson_numpy,ORL_array_poisson)

    ORL_arrayy_random = add_random_block_noise(ORL_array)
    numpy_to_pic(ORL_arrayy_random,ORL_random_block_noise_path)
    np.save(ORL_random_block_numpy,ORL_arrayy_random)

    ORL_array_salt = add_salt_and_pepper_noise(ORL_array)
    numpy_to_pic(ORL_array_salt,ORL_salt_pepper_noise_path)
    np.save(ORL_salt_numpy,ORL_array_salt)

    ORL_array_uniform = add_uniform_noise(ORL_array)
    numpy_to_pic(ORL_array_uniform,ORL_uniform_noise_path)
    np.save(ORL_uniform_numpy,ORL_array_uniform)

    # kodak24
    kodak24_array_gaussian = add_gaussian_noise(kodak24_array)
    numpy_to_pic(kodak24_array_gaussian,kodak24_gaussian_noise_path)
    np.save(kodak24_gausssian_numpy,kodak24_array_gaussian)

    kodak24_array_poisson = add_poisson_noise(kodak24_array)
    numpy_to_pic(kodak24_array_poisson,kodak24_poisson_noise_path)
    np.save(kodak24_poisson_numpy,kodak24_array_poisson)

    kodak24_arrayy_random = add_random_block_noise(kodak24_array)
    numpy_to_pic(kodak24_arrayy_random,kodak24_random_block_noise_path)
    np.save(kodak24_random_block_numpy,kodak24_arrayy_random)

    kodak24_array_salt = add_salt_and_pepper_noise(kodak24_array)
    numpy_to_pic(kodak24_array_salt,kodak24_salt_pepper_noise_path)
    np.save(kodak24_salt_numpy,kodak24_array_salt)

    kodak24_array_uniform = add_uniform_noise(kodak24_array)
    numpy_to_pic(kodak24_array_uniform,kodak24_uniform_noise_path)
    np.save(kodak24_uniform_numpy,kodak24_array_uniform)