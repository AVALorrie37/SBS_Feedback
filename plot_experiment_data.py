import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# csvframe = pd.read_csv('D:\\Documents\\项目\\1013实验数据\\1013YAP\\TRIANGLE16G120M.csv', skiprows=6, nrows=20000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1103\\0S\\12G200M.csv', skiprows=6, nrows=20000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\________.csv', skiprows=6, nrows=50000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1121部分实验结果\\50000P14G200M.csv', skiprows=6, nrows=50000)
# csvframe = pd.read_csv('D:\\Documents\\项目\\1214\\ABS01.csv', skiprows=6, nrows=20000)
csvframe2 = pd.read_csv('D:\\Documents\\项目\\1219\\2.csv', skiprows=6, nrows=20000)
csvframe3 = pd.read_csv('D:\\Documents\\项目\\1219\\3.csv', skiprows=6, nrows=20000)

# plt.plot(csvframe['Freq(Hz)'], csvframe['S21(DB)'])
# plt.xlabel("Freq(Hz)")
# plt.ylabel("S21(DB)")

def awgn_filter(x, window_size):
    length = x.size - window_size
    y = x
    for i in range(length):
        y[i] = np.sum(x[i:i+window_size])/window_size
    z = y
    for i in np.invert(range(length)):
        z[i+window_size] = np.sum(y[i:i+window_size])/window_size
    return z


# -------------画开关幅频/相频--------------
select = 1  # 幅频-0；相频-1
if select == 0:
    if 'S21(MAG)' in csvframe2:
        plt.figure(1)  # 画开关幅频（MAG)
        csvframe2_3_mag = csvframe2['S21(MAG)'] / csvframe3['S21(MAG)']
        csvframe2_3_mag = np.maximum(csvframe2_3_mag, np.ones(csvframe2_3_mag.size)*0.00001)
        csvframe2_3_mag = 10*np.log(csvframe2_3_mag)/np.log(10)
        csvframe2_3_mag = awgn_filter(np.array(csvframe2_3_mag), 50)

        # csvframe2_3_mag = awgn_filter(csvframe2_3_mag, 80)
        plt.plot(csvframe2['Freq(Hz)']/(10**9), csvframe2_3_mag)
        plt.xlabel("Freq(GHz)")
        plt.ylabel("S21(DB)")
    elif 'S21(DB)' in csvframe2:
        plt.figure(2)  # 画开关幅频（DB)
        csvframe2_3_mag = csvframe2['S21(DB)'] - csvframe3['S21(DB)']
        csvframe2_3_mag = awgn_filter(np.array(csvframe2_3_mag), 40)
        plt.plot(csvframe2['Freq(Hz)']/(10**9), csvframe2_3_mag)
        plt.xlabel("Freq(GHz)")
        plt.ylabel("S21(DB)")
elif select == 1:
    plt.figure(3)  # 画开关相频（DEG)
    csvframe2_3_deg = np.mod(csvframe2['S21(DEG)'] - csvframe3['S21(DEG)']+180, 360)-180
    # csvframe2_3_deg = csvframe2['S21(DEG)'] - csvframe3['S21(DEG)']
    csvframe2_3_deg = awgn_filter(np.array(csvframe2_3_deg), 30)
    plt.plot(csvframe2['Freq(Hz)']/(10**9), csvframe2_3_deg)
    plt.xlabel("Freq(GHz)")
    plt.ylabel("S21(DEG)")

    ## 标记最大点和最小点
    deg_max = np.max(csvframe2_3_deg)
    p1 = csvframe2['Freq(Hz)'][np.argmax(csvframe2_3_deg)]/(10**9)
    plt.text(p1, deg_max, '   '+str((float('%.2f' % p1), float('%.2f'% deg_max))),ha='left', va='top', fontsize=15)

    deg_min = np.min(csvframe2_3_deg)
    p2 = csvframe2['Freq(Hz)'][np.argmin(csvframe2_3_deg)]/(10**9)
    plt.text(p2, deg_min, str((float('%.2f' % p2), float('%.2f'% deg_min)))+'    ',ha='right', va='bottom', fontsize=15)

    plt.scatter([p1,p2], [deg_max,deg_min], s=35, marker='*', color='red')

# plt.xlim(3.12, 3.1205)


plt.show()