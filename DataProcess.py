"""
Description    :   
Author         :   LXL 
Modified Time  :   2022/10/30 20:00:51
"""

import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
# from datasets.SequenceDatasets import dataset
# from datasets.sequence_aug import *
from tqdm import tqdm
import TimeFrenqParamCal as TDC
import numpy as np

signal_size = 1024
Fs = 12000


datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
datanameM1 = ["105.mat", "118.mat", "130.mat"]  # 1797rpm&0.1778mmSlot，对应5.2.1
datanameM2 = ["105.mat", "169.mat", "209.mat"]  # 1797rpm&inner 12k 对应5.2.2

dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
dataname5 = ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
             "315.mat"]  # 1797rpm
dataname6 = ["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
             "316.mat"]  # 1772rpm
dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname9 = ["109.mat", "122.mat", "135.mat", "174.mat", "189.mat", "201.mat", "213.mat", "250.mat",
             "262.mat"]  # 1797rpm
dataname10 = ["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "251.mat",
              "263.mat"]  # 1772rpm
dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "252.mat",
              "264.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "253.mat",
              "265.mat"]  # 1730rpm
# label
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join('D:\轴承故障代码\code', root, datasetname[3])
    data_root2 = os.path.join('D:\轴承故障代码\code', root, datasetname[0])

    path1 = os.path.join('D:\轴承故障代码\code', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->`1750rpm;3->1730rpm
    data, lab = data_load(path1, axisname=normalname[0],label=0)  # nThe label for normal data is 0

    for i in tqdm(range(len(datanameM2))):
        path2 = os.path.join('D:\轴承故障代码\code', data_root2, datanameM2[i])

        data1, lab1 = data_load(path2, datanameM2[i], label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


def data_load(filename, axisname, label):
    '''
    程序只提取一个文件中的一个特征数据，如驱动端振动数据，但通过滑窗，生成多行数据，并且为这组数据生成对应的一组相同值的label。
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    data = []
    
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]/8:   #随机森林准确率太高，因此缩小样本数量，变为1/4，这个地方除了一个4
        f2 = TDC.get_time_domain_features(pd.DataFrame({"data": list(np.array(fl[start:end]).flatten())})) #loadmat得到的是一个二维数组，虽然行数组只有一个元素，要用flatten()函数来变换为一维数组。
        ff,Fy=TDC.Do_fft(list(np.array(fl[start:end]).flatten()),Fs)  #此处同上一行，需要整理为一维数组
        fDchara = TDC.get_fre_domain_features(ff, Fy)
        f2.extend(fDchara)
        data.append(f2)
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


def train_test_split_order(data_pd, test_size=0.8, num_classes=10):
    '''
        将总数据集进行拆分，分为训练集和数据集
    param:
        data_pd:总数据集
        test_size:测试集的比例，例如80%
        num_classes:标签种类数
    return:
        train_pd:训练集
        val_pd:验证集
    modified:
    2022-10-6
    '''
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes): #以label的值为分类，进行循环数据处理
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True) #提取label为i的组成新的dataset，并且通过reset_index：重新设置索引列。
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], ignore_index=True)   #将label=i的数据取出一定比例的值，作为训练集，ignore_index代表新生成的数据集index列重新标记。
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], ignore_index=True) #这是生成验证集。
    return train_pd,val_pd

def data_save_CSV(root):
    """
    Description: 
    Param: 
    Return:    
    Modified Time  :   2022/10/30 20:16:55
    """
    # 调用函数得到分组提取、并滑窗后的多组数据，分别拆分为特征数据和标签数据
    list_data = get_files(root, False)
    data_pd = pd.DataFrame(list_data[0])
    label_pd = pd.DataFrame(list_data[1])
    # 将特征数据组和标签数据组分辨保存为CSV文件，便于多次调用
    data_pd.to_csv('data_pd2.csv')   #5.2.1对应data_pd.csv,5.2.2对应data_pd2.csv
    label_pd.to_csv('label_pd2.csv') #5.2.1对应label_pd.csv,5.2.2对应label_pd2.csv


if __name__ == '__main__':
    
    data_save_CSV('CWRBearingData')