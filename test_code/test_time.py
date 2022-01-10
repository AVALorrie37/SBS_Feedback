# 根据当前日期创建文件夹
import datetime
import os

today_date = datetime.datetime.now().strftime('%Y-%m-%d')
print(today_date)

# today_time = datetime.datetime.now().strftime('%H:%M:%S')
# print(today_time)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


# 定义要创建的目录
mkpath = os.path.join(r"D:\Documents\项目", today_date)
# 调用函数
mkdir(mkpath)