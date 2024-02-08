import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from pandas.plotting import autocorrelation_plot
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.stats import maxwell

#一些没补回来的代码：画正态的（调整func_0得到）
#gmm（直接改了模型个数）
#其他不成功的分布尝试
#原始数据展示（直接爆改了主函数中注释的那段代码，因为每个科目读取的行数不一致）


#读入excel
def input(path):
    data_frame = pd.read_excel(path,engine="openpyxl")
    return data_frame

#平滑
def smooth(Data,n):
    k = len(Data)
    ret=[]
    for i in range(k):
        sum = 0
        for j in range(n):
            if i + j < k:
                if np.isnan(Data[i+j])==False:
                    sum+=Data[i + j]
        
        if np.isnan(Data[i])==False:
            ret.append(sum / n)
        else:
            ret.append(np.nan)
    return np.array(ret)

#计算拟合系数
def cal_resid(y0,y1):
    N=len(y0)
    if N!=len(y1):
        print("wrong")
    a=0.
    b=0.
    c=0.
    for i in range(N):
        a+=np.power(y0[i]-y1[i],2)
        b+=np.power(y0[i],2)
        c+=y0[i]
    return 1-np.divide(a,b-np.power(np.divide(c,N),2))

#曲线拟合所用函数（请勿轻易转变成等价形式，curve_fit中间可能会导致值溢出）
#偏态分布
def func_0(x,*p):
    alpha,loc,scale,a= p
    return alpha * skewnorm.pdf(x,a,loc,scale)
def func_1(x,*p):
    alpha1,alpha2,loc1,loc2,scale1,scale2,a1,a2= p
    p1=[alpha1,loc1,scale1,a1]
    p2=[alpha2,loc2,scale2,a2]
    return func_0(x,*p1) + func_0(x,*p2)

#类麦克斯韦分布
def func_2(x,*p):
    alpha,m,n=p
    x=np.piecewise(x,[x<=n,x>n],[n,lambda x:x])
    return alpha*np.power(x-n,2)*np.power(m,1.5)*np.exp(-1*m*np.power(x-n,2))
def func_3(x,*p):
    alpha1,alpha2,m1,m2,n1,n2=p
    p1=[alpha1,m1,n1]
    p2=[alpha2,m2,n2]
    return func_2(x,*p1)+func_2(x,*p2)

def func_4(x,*p):
    alpha1,alpha2,alpha3,m1,m2,m3,n1,n2,n3=p
    p1=[alpha1,m1,n1]
    p2=[alpha2,m2,n2]
    p3=[alpha3,m3,n3]
    return func_2(x,*p1)+func_2(x,*p2)+func_2(x,*p3)

#积分
def func_5(x,*p):
    alpha,loc,scale,a= p
    y1=10*x
    y2=10*x+10
    ret=alpha * (skewnorm.cdf(y2,a,loc,scale)-skewnorm.cdf(y1,a,loc,scale))
    return ret


def func_6(x,*p):
    alpha,gamma,beta=p
    y1=10*x
    y2=10*x+10
    if beta<0:
        n=0
    else:
        n=np.sqrt(2*beta)
    ret=alpha * np.sqrt(np.pi)*beta*n*(maxwell.cdf(n*(y1+10),n*gamma,1.)-maxwell.cdf(n*y1,n*gamma,1.))/2
    return ret

#使人数和具体x按顺序对应（原数据为了写图表的label是str的）
def process(pX,Raw):
    N = Raw.shape[0]
    x = []
    z=[]
    y = []
    t = 0
    for n in range(N):
        if np.isnan(Raw[n]) == False:
            x.append(float(t))
            z.append(pX[n])
            y.append(Raw[n])
            t = t + 1
    X = np.array(x)
    Y = np.array(y)
    Z= np.array(z)
    return X,Y,Z

#gmm
def gmm(ax,data,k):
    X,Y,Z = process(ax,data)
    N = X.shape[0]
    Data_list = []
    for i in range(N):
        J = Y[i]
        for j in range(J):
            Data_list.append(X[i])
    Data = np.array(Data_list).reshape(-1,1)
    models = []
    Index = np.arange(1,k + 1)
    for i in range(k):
        models.append(GaussianMixture(n_components=Index[i],n_init=5).fit(Data))
    AIC = [m.aic(Data) for m in models]
    M = models[np.argmin(AIC)]
    px = np.linspace(0,N,1000)
    logprob = M.score_samples(px.reshape(-1,1))
    res = M.predict_proba(px.reshape(-1,1))
    pdf = np.exp(logprob)
    pdf_ind = res * pdf[:,np.newaxis]
    return px,pdf,pdf_ind

#以下调用curve_fit时如使用了flip请勿去掉，curve_fit比较神奇,
#涉及到flip的部分，x的最大值减去位置参数所得值为最终参数值
def gd_Chinese(df):

    x_0 = df.iloc[1:116,0].to_numpy()
    y_0 = df.iloc[1:116,3].to_numpy()
    z_0 = df.iloc[1:116,1].to_numpy()
    x_1 = df.iloc[1:116,5].to_numpy()
    y_1 = df.iloc[1:116,8].to_numpy()
    z_1 = df.iloc[1:116,6].to_numpy()

    
    labels = [i for i in range(4,len(x_0),5)]
    labels.insert(0,0)

    plt.figure()
    plt.xticks(labels,rotation=90)
    plt.plot(x_0,y_0,color="blue",label="物理类")
    plt.plot(x_0,y_1,color="red",label="历史类")
    plt.legend()
    
    y_2 = smooth(y_0,5)
    y_3 = smooth(y_1,5)
    plt.figure()
    plt.xticks(labels,rotation=90)
    plt.plot(x_0,y_2,color="blue",label="物理类")
    plt.plot(x_0,y_3,color="red",label="历史类")
    plt.legend()

    px,pdf,pdf_ind=gmm(x_0,z_0,5)
    plt.figure()
    plt.plot(y_2,color="blue",label="data")
    plt.plot(px,pdf,color="red",label="fit")
    plt.plot(px,pdf_ind,color="red",linestyle="--")
    plt.legend()

    px,pdf,pdf_ind=gmm(x_1,z_1,5)
    plt.figure()
    plt.plot(y_3,color="blue",label="data")
    plt.plot(px,pdf,color="red",label="fit")
    plt.plot(px,pdf_ind,color="red",linestyle="--")
    plt.legend()

    p=[1.,0.,1.,1.]
    x,y,z = process(x_0,y_2)
    popt,pcov=curve_fit(func_0,x,y,p0 = p,maxfev=5000)
    py_1=func_0(x,*popt)
    print(cal_resid(y,py_1))
    plt.figure()
    labels = [i for i in range(4,len(z),5)]
    labels.insert(0,0)
    plt.xticks(labels,rotation=90)
    plt.plot(z,y,color="blue",label="data(smooth)")
    plt.plot(py_1,color="red",label="fit",linestyle="--")
    plt.legend()

    #偏态分布对历史类的一次尝试
    P=[1.,0.,1.,1.]
    X,Y,z = process(x_1,y_3)
    popt,pcov=curve_fit(func_0,X,Y,p0 = P,maxfev=5000)
    pY=func_0(X,*popt)
    plt.figure()
    labels = [i for i in range(4,len(z),5)]
    labels.insert(0,0)
    plt.xticks(labels,rotation=90)
    plt.plot(z,Y,color="blue",label="data(smooth)")
    plt.plot(pY,color="red",label="fit",linestyle="--")
    plt.legend()

    p=[1.,1.,1.]
    x,y,z = process(x_1,y_3)
    popt,pcov=curve_fit(func_2,x,np.flip(y),p0 = p,maxfev=5000)
    py_2=np.flip(func_2(x,*popt))
    plt.figure()
    labels = [i for i in range(4,len(z),5)]
    labels.insert(0,0)
    plt.xticks(labels,rotation=90)
    plt.plot(z,y,color="blue",label="data(smooth)")
    plt.plot(py_2,color="red",label="fit",linestyle="--")
    plt.legend()

    plt.figure()
    plt.plot(py_1,color="blue",label="物理类",linestyle="--")
    plt.plot(py_2,color="red",label="历史类",linestyle="--")
    plt.legend()


def gd_Math(df):
    x_0 = df.iloc[1:136,0].to_numpy()
    y_0 = df.iloc[1:136,3].to_numpy()
    z_0 = df.iloc[1:136,1].to_numpy()
    x = df.iloc[1:136,4].to_numpy()
    x_1 = df.iloc[1:136,5].to_numpy()
    y_1 = df.iloc[1:136,8].to_numpy()
    z_1 = df.iloc[1:136,6].to_numpy()
    
    labels = [i for i in range(4,len(x),5)]
    labels.insert(0,0)

    plt.figure()
    plt.xticks(labels,rotation=90)
    plt.plot(x,y_0,color="blue",label="物理类")
    plt.plot(x,y_1,color="red",label="历史类")
    plt.legend()

    y_2 = smooth(y_0,5)
    y_3 = smooth(y_1,5)
    plt.figure()
    plt.xticks(labels,rotation=90)
    plt.plot(x,y_2,color="blue",label="物理类")
    plt.plot(x,y_3,color="red",label="历史类")
    plt.legend()

    
    px,pdf,pdf_ind=gmm(x_0,z_0,1)
    plt.figure()
    plt.plot(y_2,color="blue",label="data")
    plt.plot(px,pdf,color="red",label="fit")
    plt.plot(px,pdf_ind,color="red",linestyle="--")
    plt.legend()

    px,pdf,pdf_ind=gmm(x_1,z_1,1)
    plt.figure()
    plt.plot(y_3,color="blue",label="data")
    plt.plot(px,pdf,color="red",label="fit")
    plt.plot(px,pdf_ind,color="red",linestyle="--")
    plt.legend()

    p=[0.8,0.2,0.,0.,1.,1.,1.,1.]
    x,y,z = process(x_0,y_2)
    popt,pcov=curve_fit(func_1,x,y,p0 = p,maxfev=5000)
    p1=popt[[0,2,4,6]]
    p2=popt[[1,3,5,7]]
    py_1=func_1(x,*popt)
    f1=func_0(x,*p1)
    f2=func_0(x,*p2)
    print(cal_resid(y,py_1))
    plt.figure()
    labels = [i for i in range(4,len(z),5)]
    labels.insert(0,0)
    plt.xticks(labels,rotation=90)
    plt.plot(z,y,color="blue",label="data(smooth)")
    plt.plot(py_1,color="red",label="fit",linestyle="--")
    plt.plot(f1,color="red",linestyle="--")
    plt.plot(f2,color="red",linestyle="--")
    plt.legend()

    p=[0.5,0.5,1.,1.,0.,0.]
    x,y,z = process(x_1,y_3)
    print(len(y_3))
    popt,pcov=curve_fit(func_3,x,y,p0 = p,maxfev=5000)
    py_2=func_3(x,*popt)
    p1=popt[[0,2,4]]
    p2=popt[[1,3,5]]
    f3=func_2(x,*p1)
    f4=func_2(x,*p2)
    plt.figure()
    labels = [i for i in range(4,len(z),5)]
    labels.insert(0,0)
    plt.xticks(labels,rotation=90)
    plt.plot(z,y,color="blue",label="data(smooth)")
    plt.plot(py_2,color="red",label="fit",linestyle="--")
    plt.plot(f3,color="red",linestyle="--")
    plt.plot(f4,color="red",linestyle="--")
    plt.legend()

    plt.figure()
    plt.plot(py_1,color="blue",label="物理类",linestyle="--")
    plt.plot(f1,color="blue",linestyle="--")
    plt.plot(f2,color="blue",linestyle="--")
    plt.plot(py_2,color="red",label="历史类",linestyle="--")
    plt.plot(f3,color="red",linestyle="--")
    plt.plot(f4,color="red",linestyle="--")
    plt.legend()
    

def gd_English(df):
    x_0 = df.iloc[1:136,0].to_numpy()
    y_0 = df.iloc[1:136,3].to_numpy()
    z_0 = df.iloc[1:136,1].to_numpy()
    x_1 = df.iloc[1:136,5].to_numpy()
    y_1 = df.iloc[1:136,8].to_numpy()
    z_1 = df.iloc[1:136,6].to_numpy()
    
    labels = [i for i in range(4,len(x_1),5)]
    labels.insert(0,0)

    plt.figure()
    plt.xticks(labels,rotation=90)
    plt.plot(x_1,y_0,color="blue",label="物理类")
    plt.plot(x_1,y_1,color="red",label="历史类")
    plt.legend()

    y_2 = smooth(y_0,5)
    y_3 = smooth(y_1,5)
    plt.figure()
    plt.xticks(labels,rotation=90)
    plt.plot(x_1,y_2,color="blue",label="物理类")
    plt.plot(x_1,y_3,color="red",label="历史类")
    plt.legend()

    
    
    X,Y,Z = process(x_0,y_0)
    sd=seasonal_decompose(Y,period=15,model='additive')
    plt.figure()
    plt.plot(sd.observed,color="blue",label="observed")
    plt.plot(sd.trend,color="red",linestyle="--",label="trend")
    plt.legend()
    y_4=sd.trend
    plt.figure()
    plt.plot(sd.resid)

    X,Y,Z = process(x_1,y_1)
    sd=seasonal_decompose(Y,period=15,model='additive')
    plt.figure()
    plt.plot(sd.observed,color="blue",label="observed")
    plt.plot(sd.trend,color="red",linestyle="--",label="trend")
    plt.legend()
    y_5=sd.trend
    plt.figure()
    plt.plot(sd.resid)
    

    P=[0.8,0.2,0.,0.,1.,1.,1.,1.]
    X,Y,Z = process(x_0,y_4)
    popt,pcov=curve_fit(func_1,X,Y,p0 = P,maxfev=5000)
    p1=popt[[0,2,4,6]]
    p2=popt[[1,3,5,7]]
    py_1=func_1(X,*popt)
    f3=func_0(X,*p1)
    f4=func_0(X,*p2)
    print(cal_resid(Y,py_1))
    plt.figure()
    plt.plot(Y,color="blue",label="data(smooth)")
    plt.plot(py_1,color="red",label="fit",linestyle="--")
    plt.plot(f3,color="red",linestyle="--")
    plt.plot(f4,color="red",linestyle="--")
    plt.legend()


    P=[0.4,0.3,0.3,1.,1.,1.,0.,0.,0.]
    X,Y,Z = process(x_1,y_5)
    popt,pcov=curve_fit(func_4,X,Y,p0 = P,maxfev=5000)
    py_2=func_4(X,*popt)
    p1=popt[[0,3,6]]
    p2=popt[[1,4,7]]
    p3=popt[[2,5,8]]
    f1=func_2(X,*p1)
    f2=func_2(X,*p2)
    f3=func_2(X,*p3)
    print(cal_resid(Y,py_2))
    plt.figure()
    plt.plot(Y,color="blue",label="data(smooth)")
    plt.plot(py_2,color="red",label="fit",linestyle="--")
    plt.plot(f3,color="blue",linestyle="--")
    plt.plot(f4,color="blue",linestyle="--")
    plt.plot(f1,color="red",linestyle="--")
    plt.plot(f2,color="red",linestyle="--")
    plt.plot(f3,color="red",linestyle="--")
    plt.legend()

    plt.figure()
    plt.plot(py_1,color="blue",label="物理类",linestyle="--")
    plt.plot(py_2,color="red",label="历史类",linestyle="--")
    plt.plot(f1,color="red",linestyle="--")
    plt.plot(f2,color="red",linestyle="--")
    plt.plot(f3,color="red",linestyle="--")
    plt.legend()

    
def fj_Math(df_0,df_1):
    x_0 = df_0.iloc[1:16,0].to_numpy()
    y_0 = df_0.iloc[1:16,2].to_numpy()
    z_0 = df_0.iloc[1:16,1].to_numpy()
    x_1 = df_0.iloc[1:16,4].to_numpy()
    y_1 = df_0.iloc[1:16,6].to_numpy()
    z_1 = df_0.iloc[1:16,5].to_numpy()
    
    x_2 = df_1.iloc[1:16,0].to_numpy()
    y_2 = df_1.iloc[1:16,2].to_numpy()
    x_3 = df_1.iloc[1:16,4].to_numpy()
    y_3 = df_1.iloc[1:16,6].to_numpy()

    plt.figure()
    plt.xticks(rotation=30)
    plt.plot(x_0,y_0,color="blue",label="21年物理类",linestyle="--")
    plt.plot(x_0,y_1,color="red",label="21年历史类",linestyle="--")
    plt.plot(x_0,y_2,color="blue",label="22年物理类")
    plt.plot(x_0,y_3,color="red",label="22年历史类")
    plt.legend()

    plt.figure()
    plt.xticks(rotation=30)
    plt.plot(x_0,smooth(y_0,3),color="blue",label="21年物理类",linestyle="--")
    plt.plot(x_0,smooth(y_1,3),color="red",label="21年历史类",linestyle="--")
    
    plt.plot(x_0,smooth(y_2,3),color="blue",label="22年物理类")
    plt.plot(x_0,smooth(y_3,3),color="red",label="22年历史类")
    plt.legend()

    p=[1.,0.,10.,1.]
    x,y,z=process(x_0,y_0)
    popt,pcov=curve_fit(func_5,x,y,p0=p,maxfev=5000)
    py_0=func_5(x,*popt)
    print(popt)
    print(cal_resid(y,py_0))
    plt.figure()
    plt.plot(y,color="blue",label="data(smooth)")
    plt.plot(py_0,color="red",label="fit",linestyle="--")
    plt.legend()

    p=[1.,0.,10.,1.]
    x,y,z=process(x_2,y_2)
    popt,pcov=curve_fit(func_5,x,y,p0=p,maxfev=5000)
    py_2=func_5(x,*popt)
    print(cal_resid(y,py_2))
    print(popt)
    plt.figure()
    plt.plot(y,color="blue",label="data(smooth)")
    plt.plot(py_2,color="red",label="fit",linestyle="--")
    plt.legend()

    #这个地方要试，如果换成1.,0.1.，需要增加搜索次数maxfev
    p=[4000.,0.,0.0007]
    x,y,z=process(x_1,y_1)
    popt,pcov=curve_fit(func_6,x,np.flip(y),p0=p,maxfev=20000)
    py_1=np.flip(func_6(x,*popt))
    print(cal_resid(y,py_1))
    print(popt)
    plt.figure()
    plt.plot(y,color="blue",label="data(smooth)")
    plt.plot(py_1,color="red",label="fit",linestyle="--")
    plt.legend()

    p=[1.,0.,1.]
    x,y,z=process(x_3,y_3)
    popt,pcov=curve_fit(func_6,x,y,p0=p,maxfev=5000)
    py_3=func_6(x,*popt)
    print(cal_resid(y,py_3))
    plt.figure()
    plt.plot(y,color="blue",label="data(smooth)")
    plt.plot(py_3,color="red",label="fit",linestyle="--")
    plt.legend()

    plt.figure()
    plt.plot(py_0,color="blue",label="物理类",linestyle="--")
    plt.plot(py_1,color="red",label="历史类",linestyle="--")
    plt.legend()

    plt.figure()
    plt.plot(py_2,color="blue",label="物理类",linestyle="--")
    plt.plot(py_3,color="red",label="历史类",linestyle="--")
    plt.legend()


        
#最好一组组来，不要同时运行语文数学英语的函数
if __name__ == "__main__":
    pd.set_option('display.max_rows',None) 
    pd.set_option('display.max_columns',None) 
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    #需要改一下路径
    path_list = [r'高考数据\广东\21\语文.xlsx',
                 r'高考数据\广东\21\数学.xlsx',
                 r'高考数据\广东\21\英语.xlsx',
                 r'高考数据\福建\21\数学.xlsx',
                 r'高考数据\福建\22\数学.xlsx',
                 r'高考数据\广东\21\语文 - raw.xlsx',
                 r'高考数据\广东\21\数学 - raw.xlsx',
                 r'高考数据\广东\21\英语 - raw.xlsx',]
    

    df_0 = input(path_list[0])
    
    #gd_Chinese(df_0)


    df_1 = input(path_list[1]) 
    gd_Math(df_1)

    df_2 = input(path_list[2])
    #gd_English(df_2)


    df_3 = input(path_list[3])
    df_4 = input(path_list[4])
    #fj_Math(df_3,df_4)

    #df=input(path_list[5])
    #x_0 = df.iloc[1:117,0].to_numpy()
    #y_0 = df.iloc[1:117,3].to_numpy()
    #x_1 = df.iloc[1:117,5].to_numpy()
    #y_1 = df.iloc[1:117,8].to_numpy()
    #print(x_0)
    
    #labels = [i for i in range(4,len(x_0),5)]
    #labels.insert(0,0)
    #labels[-1]=len(x_0)-1
    #m=len(x_0)
    #for n in range(m):
    #    x_0[n]=str(x_0[n])
    #plt.figure()
    #plt.xticks(labels,rotation=90)
    #plt.plot(x_0,y_0,color="blue")

    plt.tight_layout(w_pad=1.0)
    plt.show()
