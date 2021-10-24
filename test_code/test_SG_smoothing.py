import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    csvframe2 = pd.read_csv('D:\\Documents\\项目\\2021-9-22 扫频验证\\300MHz正常+1-340ms.csv')
    csvframe3 = pd.read_csv('D:\\Documents\\项目\\2021-9-22 扫频验证\\300MHz-BJ.csv')
    csvframe2_3_mag = csvframe2['y0000'] - csvframe3['y0000']

    csvframe2_3_mag = savgol_filter(csvframe2_3_mag, 301, 3)

    plt.plot(csvframe2['x0000'] / (10 ** 9), csvframe2_3_mag)
    plt.xlabel("Freq(GHz)")
    plt.ylabel("S21(DB)")
    plt.show()