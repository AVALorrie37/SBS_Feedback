import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# csvframe = pd.read_csv('D:\\Documents\\项目\\1013实验数据\\1013YAP\\TRIANGLE16G120M.csv', skiprows=6, nrows=20000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1103\\0S\\12G200M.csv', skiprows=6, nrows=20000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\________.csv', skiprows=6, nrows=50000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\50000P14G200M.csv', skiprows=6, nrows=50000)
csvframe = pd.read_csv('D:\\Documents\\项目\\1214\\ABS01.csv', skiprows=6, nrows=20000)
csvframe2 = pd.read_csv('D:\\Documents\\项目\\1219\\18.csv', skiprows=6, nrows=80000)
csvframe3 = pd.read_csv('D:\\Documents\\项目\\1219\\19.csv', skiprows=6, nrows=80000)

# plt.plot(csvframe['Freq(Hz)'], csvframe['S21(DB)'])
# plt.xlabel("Freq(Hz)")
# plt.ylabel("S21(DB)")

# plt.figure(1)  # 画开关幅频（MAG)
# csvframe2_3_mag = csvframe2['S21(MAG)'] / csvframe3['S21(MAG)']
# csvframe2_3_mag = np.maximum(csvframe2_3_mag, np.ones(csvframe2_3_mag.size)*0.00001)
# csvframe2_3_mag = 10*np.log(csvframe2_3_mag)/np.log(10)
#
# plt.plot(csvframe2['Freq(Hz)']/(10**9), csvframe2_3_mag)
# plt.xlabel("Freq(GHz)")
# plt.ylabel("S21(DB)")

# plt.figure(2)  # 画开关幅频（DB)
# csvframe2_3_mag = csvframe2['S21(DB)'] - csvframe3['S21(DB)']
# plt.plot(csvframe2['Freq(Hz)']/(10**9), csvframe2_3_mag)
# plt.xlabel("Freq(GHz)")
# plt.ylabel("S21(DB)")

plt.figure(3)  # 画开关相频（DEG)
csvframe2_3_deg = np.mod(csvframe2['S21(DEG)'] - csvframe3['S21(DEG)']+180, 360)-180
# csvframe2_3_deg = csvframe2['S21(DEG)'] - csvframe3['S21(DEG)']
plt.plot(csvframe2['Freq(Hz)']/(10**9), csvframe2_3_deg)
plt.xlabel("Freq(GHz)")
plt.ylabel("S21(DEG)")

# plt.xlim(3.32, 3.34)

plt.show()