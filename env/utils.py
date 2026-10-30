import math

import numpy as np


def softmax(x):
    """
    x: 输入的向量或矩阵（支持 batch）
    return: softmax 输出，形状与 x 相同
    """
    # 如果是二维的，按行做 softmax
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)  # 避免数值溢出
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        x = x - np.max(x)  # 避免数值溢出
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


def haversine_coord(p1, p2, km=True):
    if p1 == p2: return 0.0

    lat1, lon1 = p1
    lat2, lon2 = p2
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) * math.cos(lat1) * math.cos(lat2))
    c = 2 * math.asin(math.sqrt(a))
    if not km: return 6371 * c * 1000.0
    return 6371 * c


def haversine(p1, p2, km=True):
    return haversine_coord(p1=(p1['y'], p1['x']), p2=(p2['y'], p2['x']), km=km)


def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1-p2, 2)))


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def compute_area(data, theta):
    all_areas = []
    all_values = []
    for j_ in range(data.shape[1]):
        values = np.concatenate([data[0, j_], [data[0, j_][0]]])
        x = values[:-1] * np.cos(theta[:-1])
        y = values[:-1] * np.sin(theta[:-1])
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        all_areas.append(area)
        all_values.append(values)
    return all_areas, all_values
