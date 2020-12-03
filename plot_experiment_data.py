import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# csvframe = pd.read_csv('D:\\Documents\\项目\\1013实验数据\\1013YAP\\TRIANGLE16G120M.csv', skiprows=6, nrows=20000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1103\\0S\\12G200M.csv', skiprows=6, nrows=20000)
csvframe = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\________.csv', skiprows=6, nrows=50000)
csvframe = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\50000P14G200M.csv', skiprows=6, nrows=50000)

plt.plot(csvframe['Freq(Hz)'], csvframe['S21(DB)'])
plt.xlabel("Freq(Hz)")
plt.ylabel("S21(DB)")
plt.show()