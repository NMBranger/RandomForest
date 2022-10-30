"""
Description    :   计算振动信号的时域、频域、时频域特征值
Author         :   LXL 
Modified Time  :   2022/10/30 21:29:05
"""
import numpy as np
import math
import numpy as np
from PyEMD import EEMD, EMD, Visualisation

def get_time_domain_features(data):
    """
    函数说明:计算振动信号的11个时域特征 
    Parameters:
        data:一维振动信号
    Returns:
        fea:11个时域特征组成的数组
    modified:
        2022-10-4
    """
    x_rms = 0
    absXbar = 0
    x_r = 0
    S = 0
    K = 0
    k = 0
    x_rms = 0
    fea = []
    len_ = len(data.loc[:, 'data'])
    # print(data.loc[:, 'data'])
    mean_ = data.mean()  # 1.均值
    var_ = data.var()  # 2.方差
    std_ = data.std()  # 3.标准差
    max_ = data.max()  # 4.最大值
    min_ = data.min()  # 5.最小值
    x_p = max(abs(max_[0]), abs(min_[0]))  # 6.峰值
    for i in range(len_):
        x_rms += data.loc[i, 'data'] ** 2
        absXbar += abs(data.loc[i, 'data'])
        x_r += math.sqrt(abs(data.loc[i, 'data']))
        S += (data.loc[i, 'data'] - mean_[0]) ** 3
        K += (data.loc[i, 'data'] - mean_[0]) ** 4
    x_rms = math.sqrt(x_rms / len_)  # 7.均方根值
    absXbar = absXbar / len_  # 8.绝对平均值
    x_r = (x_r / len_) ** 2  # 9.方根幅值
    W = x_rms / mean_[0]  # 10.波形指标
    C = x_p / x_rms  # 11.峰值指标
    I = x_p / mean_[0]  # 12.脉冲指标
    L = x_p / x_r  # 13.裕度指标
    S = S / ((len_ - 1) * std_[0] ** 3)  # 14.偏斜度
    K = K / ((len_ - 1) * std_[0] ** 4)  # 15.峭度

    fea = [mean_[0],absXbar,var_[0],std_[0],x_r,x_rms,x_p,max_[0],min_[0],W,C,I,L,S,K]
    
    return fea

def nextpow2(x):
    if x == 0:
        return 0 
    else:
        # print('x=', x)
        # print("log2x=", np.log2(x))
        return int(np.ceil(np.log2(x)))

def Do_fft(sig,Fs):#输入信号和采样频率
    """
    函数说明:求FFT变换 
    Parameters:
        sig:时域信号序列
        y:采样频率
    Returns:
        f:频域频率序列
        yf:频域幅值序列
    modified:
        2022-10-4
    """    
    xlen = len(sig)
    sig = sig - np.mean(sig)
    # print('N=', nextpow2(xlen))
    # NFFT = 2**nextpow2(xlen)
    NFFT = xlen
    yf = np.fft.fft(sig,NFFT)/xlen*2
    yf = abs(yf[0:int(NFFT/2+1)])
    f = Fs/2*np.linspace(0,1,int(NFFT/2+1))
    f = f[:]
    return f,yf
    #频域离散值的序号

def get_fre_domain_features(f,y):
    """
    函数说明:计算13个振动信号的频域特征 
    Parameters:
        f:频率点序列
        y:频率点对应的幅值
    Returns:
        p:13个频域特征组成的数组
    modified:
        2022-10-4
    """
    fre_line_num = len(y)
    p1 = y.mean()
    p2 = np.sqrt(np.sum((y-p1)**2)/fre_line_num)
    p3 = np.sum((y-p1)**3)/(fre_line_num*p2**3)
    p4 = np.sum((y-p1)**4)/(fre_line_num*p2**4)
    p5 = np.sum(f*y)/np.sum(y)
    p6 = np.sqrt(np.sum((f-p5)**2*y)/fre_line_num)
    p7 = np.sqrt(np.sum(f**2*y)/np.sum(y))
    p8 = np.sqrt(np.sum(f**4*y)/np.sum(f**2*y))
    p9 = np.sum(f**2*y)/np.sqrt(np.sum(y)*np.sum(f**4*y))
    p10 = p6/p5
    p11 = np.sum((f-p5)**3*y)/(p6**3*fre_line_num)
    p12 = np.sum((f-p5)**4*y)/(p6**4*fre_line_num)
    p13 = np.sum(abs(f-p5)*y)/(np.sqrt(p6)*fre_line_num)
    p = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]
    return p

def get_TF_domain_features(sig, ff):
    """
    函数说明:求时频域的特征值 
    Parameters:
        sig:时域信号序列，list类型
        ff:采样频率
    Returns:
        pp:时频域的特征值序列，list类型

    modified:
        2022-10-7
    """ 
    numb = len(sig)
    t = np.linspace(0, (numb-1)/ff, numb)
    # EEMD设置
    max_imf = 4  #设定分解出来的IMF的个数，最后画出来的话，会有max_imf+1个曲线，注意最后一个曲线时残差，设定的曲线数不同，则残差曲线就不同。
    # EEMD计算
    eemd = EEMD()
    eemd.trials = 50
    # eemd.noise_seed(12345)

    E_IMFs = eemd.eemd(sig, t, max_imf)
    imfNo = E_IMFs.shape[0]
    E = [0]*(imfNo-1)
    for num in range(imfNo-1):
        SIMF = np.square(E_IMFs[num])
        E[num] = np.sum(SIMF)
    Es = np.sum(E)
    EE = np.array(E)
    p = EE / Es 
    Hen = 0
    for num in range(imfNo-1):
        Hen += p[num]*np.log10(p[num])
    pp = p.tolist()
    pp.append(Hen)
    return pp

