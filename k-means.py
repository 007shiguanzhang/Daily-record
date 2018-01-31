# coding:utf-8
import codecs
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# loading data from text
def load_txt(path="testSet.txt"):
    x, y = [], []
    with codecs.open(path, "r", "utf-8") as f:
        for temp in f.readlines():
            temp_list = [float(i) for i in temp.split()]
            x.append(temp_list[0])
            y.append(temp_list[1])
    return x, y


# plt画图显示出来
def show_data(data, cores):
    x_list, y_list, c_list = [], [], []
    colors = 'bgrc'
    center_index = list(data.keys())
    plt.clf()
    plt.cla()
    # plt参数设置
    plt.ion()
    plt.title("动态散点图")
    plt.grid(True)
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    for core in cores:
        plt.scatter(core[0], core[1], c="m", marker="*")
    plt.pause(0.5)
    # plt的数据点
    for key, values in data.items():
        for value in values:
            plt.scatter(value[0], value[1], c=colors[center_index.index(key)], marker="o")
            plt.pause(0.001)
    #         x_list.append(value[0])
    #         y_list.append(value[1])
    #         c_list.append(colors[center_index.index(key)])
    # plt.scatter(x_list, y_list, c=c_list, marker="o")
    plt.pause(1)
    plt.plot()
    plt.ioff()
    return None


def get_link(points, cores):
    """
    :param points: 数据点list
    :param cores:  质点list
    :return:  聚类结果，更新之后的质点list
    """
    result = dict()
    for point in points:
        min_dis = None
        min_core = None
        for core in cores:
            dis = np.linalg.norm(np.array(point)-np.array(core))
            if min_dis and dis < min_dis:
                min_dis = dis
                min_core = core
            if not min_dis:
                min_dis = dis
                min_core = core
        temp = result.get(str(min_core), [])
        temp.append(point)
        result[str(min_core)] = temp
    renew_cores = []
    for core in cores:
        x_core, y_core = 0, 0
        classified_data = result.get(str(core), [])
        for p in classified_data:
            x_core += p[0]
            y_core += p[1]
        if classified_data.__len__():
            x_core = x_core/classified_data.__len__()
            y_core = y_core/classified_data.__len__()
        renew_cores.append([x_core, y_core])
    return result, renew_cores


def k_means(data, n):
    # 数据点
    x_list, y_list = data
    points = [[x, y] for x, y in zip(x_list, y_list)]
    # init
    cores = random.sample(points, n)
    last_result = None
    # start running
    result, renew_cores = get_link(points, cores)
    show_data(result, cores)
    print("start train")
    i = 0
    # loop until result unchanged
    while last_result != result and i < 100:
        print("steps:", i)
        i += 1
        last_result = result
        result, renew_cores = get_link(points, renew_cores)
        show_data(result, renew_cores)
    return result


if __name__ == "__main__":
    data = load_txt()
    result = k_means(data, 4)
