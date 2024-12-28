import os
import random

def setup_seed(seed):
    random.seed(seed)
setup_seed(20)

b = 0
dir = 'F:/Machine learning/飞机型号识别/飞机型号识别/数据集/'
files = os.listdir(dir)     
files.sort()

train = open('./train.txt', 'w')
test = open('./val.txt', 'w')
a = 0
a1 = 0
files = os.listdir('F:/Machine learning/飞机型号识别/飞机型号识别/数据集')
print(f"找到 {len(files)} 个文件夹。")
while(b < len(files)):
    print(f"处理文件夹: {files[b]}")
    label = a 
    ss = 'F:/Machine learning/飞机型号识别/飞机型号识别/数据集/' + str(files[b]) + '/' 
    pics = os.listdir(ss) 
    print(f"找到 {len(pics)} 张图片。")

    if len(pics) == 0:
        print(f"警告: 文件夹 {files[b]} 中没有图片。")

    i = 1
    train_percent = 0.8 

    num = len(pics) 
    list = range(num)  
    train_num = int(num * train_percent)  
    train_sample = random.sample(list, train_num) 
    test_num = num - train_num    

    for i in list:  
        name = str(dir) + str(files[b]) + '/' + pics[i] + ' ' + str(int(label)) + '\n'  
        if i in train_sample:  
            train.write(name)  
        else:
            test.write(name)  
    a = a + 1
    b = b + 1
train.close()  
test.close()
print("训练集和验证集文件已创建。")
