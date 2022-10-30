'''
读取csv文件,并且形成随机森林，可视化。
'''
import pandas as pd
from sklearn.model_selection import train_test_split	
from sklearn.ensemble import RandomForestClassifier	
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import tree

# datafile = 'data_pd3.csv' #5.2.3无时频域特征的训练过程用这个代码，如果做有时频域的注释掉本行，用下一行代码
datafile = 'dataTF_pd3.csv' 

def RF_training(datafile):
    """
    Description: 从csv文件提取数据，进行随机森林学习，决策树数量和深度均为5，得到套内和套外准确率，打印出来
    Param: 
        datafile: 数据文件全名
    Return:    
    Modified Time  :   2022/10/30 20:25:27
    """
    data_pd = pd.read_csv(datafile,index_col=0)  #注意：不加index_col=0的话，用values取值时，会把第一列也取为数值。
    label_pd = pd.read_csv('label_pd3.csv',index_col=0)  #注意5.2.3对应label_pd3.csv和data_pd2.csv.
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data_pd, label_pd, test_size=0.25)

    X_train = X_train1.values
    X_test = X_test1.values
    y_train = np.ravel(y_train1.values)
    y_test = np.ravel(y_test1.values)
    print(X_train[:3])
    print(y_train)
    print(np.ravel(y_train))
    print(len(y_train))
    RF = RandomForestClassifier( n_estimators=5 ,max_depth=5)	#设定决策树数量和决策树深度
    RF.fit( X_train, y_train )
    print ( "RF - Accuracy (Train):  %.4g" % 	
        metrics.accuracy_score(y_train, RF.predict(X_train)) )	
    print ( "RF - Accuracy (Test):  %.4g" % 	
        metrics.accuracy_score(y_test, RF.predict(X_test)) )
    return RF, X_train, y_train

######################################################
def Accuracy_curve(X_train, y_train, datafile, estM):
    """
    Description: 绘制决策树数量与准确率的关系曲线，将
    Param: 
        X_train, y_train, datafile, estM: 特征参数，标签参数，数据文件名称, 决策树最大数量
    Return:    
    Modified Time  :   2022/10/30 20:39:43
    """
    superpa = []
    for i in range(estM):
        rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
        rfc_s = cross_val_score(rfc,X_train,y_train,cv=10).mean()
        superpa.append(1-rfc_s)
        print(i)
    print(max(superpa),superpa.index(max(superpa)))
    plt.figure(figsize=[6,5])
    plt.plot(range(1,estM+1),superpa)
    accuName = datafile.split(".")[0] + '_accuracy.png'
    plt.savefig(accuName,dpi=600)  #对应5.2.3
    # plt.savefig('accuracy2.png',dpi=600)  #对应5.2.2
    plt.show()
    # np.savetxt("5.2.2_OOBerror.txt", [range(1,estM+1),superpa])
    textname = datafile.split(".")[0] + '.txt'

    np.savetxt(textname, [range(1,estM+1),superpa])
    # print(superpa)

###########################
def RF_Visual(X_train, RF, datafile):
    """
    Description: 绘制已知决策树RF
    Param: 
    Return:    
    Modified Time  :   2022/10/30 20:50:38
    """
    if len(X_train[0]) == 28:
        fn=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','a','b','c','d','e','f','g','h','i','j','k','l','m']
    else:
        fn=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','a','b','c','d','e','f','g','h','i','j','k','l','m','tf1','tf2','tf3','tf4','tf5']
    cn=['load 0','load 1','load 2','load 3']
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(RF.estimators_[0],
        feature_names = fn, 
        class_names=cn,
        filled = True);
    treefileName = datafile.split(".")[0] + '_tree.png'
    fig.savefig(treefileName) #5.2.3对应GraphicTree3

if __name__ == '__main__':
    RF, X_train, y_train = RF_training(datafile)
    RF_Visual(X_train, RF, datafile)
    Accuracy_curve(X_train, y_train, datafile, 50)