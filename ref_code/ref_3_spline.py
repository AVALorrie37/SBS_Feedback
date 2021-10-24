import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#数据准备
X=np.arange(-np.pi,np.pi,1) #定义样本点X，从-pi到pi每次间隔1
Y= np.sin(X)#定义样本点Y，形成sin函数
new_x=np.arange(-np.pi,np.pi,0.1) #定义差值点

#进行样条差值
import scipy.interpolate as spi

#进行一阶样条插值
ipo1=spi.splrep(X,Y,k=1) #样本点导入，生成参数
iy1=spi.splev(new_x,ipo1) #根据观测点和样条参数，生成插值

#进行三次样条拟合
ipo3=spi.splrep(X,Y,k=3) #样本点导入，生成参数
iy3=spi.splev(new_x,ipo3) #根据观测点和样条参数，生成插值



##作图
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,12))

ax1.plot(X,Y,'o',label='样本点')
ax1.plot(new_x,iy1,label='插值点')
ax1.set_ylim(Y.min()-1,Y.max()+1)
ax1.set_ylabel('指数')
ax1.set_title('线性插值')
ax1.legend()

ax2.plot(X,Y,'o',label='样本点')
ax2.plot(new_x,iy3,label='插值点')
ax2.set_ylim(Y.min()-1,Y.max()+1)
ax2.set_ylabel('指数')
ax2.set_title('三次样条插值')
ax2.legend()

plt.show()
