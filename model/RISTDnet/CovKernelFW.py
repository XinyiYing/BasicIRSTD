import numpy as np

kernels_all = {}  # key:e->kind_kernels;values:[]->kernels
num_cycle = [1, 2, 3, 4, 5]  #


def GenerateKernels():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle:  # 第i种卷积核
        kernels = {}
        for j in range(i):  # 第i种卷积核的第j个卷积核
            k_size = (2 * i) + 1  # 卷积核的尺寸
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  # 生成卷积核
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2  # 红色区域左上角的x y轴索引
            red_size = (j + 1) * 2 - 1  # 红色区域尺寸
            # 给中间红色区域值
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  # 红色区域填充值
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val  # 赋值
            # 给左、右、上、下蓝色区域赋值
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size)  # 蓝色区域填充值
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[
                                                                                 :,
                                                                                 lt_y + red_size:] = blue_val  # 赋值
            # 添加到第i中卷积核中
            kernels[j + 1] = kernel
        # 添加到所有卷积核中
        kernels_all[i] = kernels
        pass


# 生成卷积核
GenerateKernels()


def get_kernels(kind):
    """
    获取某种卷积核的所有卷积核
    :param kind: 卷积核种类 1~5
    :return: [kernels of kind]
    """
    try:
        return list(kernels_all[kind].values())
    except KeyError:
        print('下标不对！')
