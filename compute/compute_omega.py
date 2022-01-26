import os
import numpy as np
import openpyxl
import pandas as pd

path = 'D:\Projects\pythonProject\F-MADDPG\compute'
filenames = []
reward = []
omegas = []

for file in os.listdir(path):
    if os.path.splitext(file)[1] == '.txt':
        filenames.append(file.split('.')[0])
        with open(file, encoding='utf-8') as f:
            datas = []
            for line in f:
                # print(line)
                string = line.split(':')
                # print(string[-1])
                if len(string) > 2:
                    datas.append(string[-1].split('\n')[0])
            omegas = [float(s) for s in datas[1::2]]
            rewards = [float(s) for s in datas[0::2]]
            print(len(omegas), len(rewards))

# 输出到Excel
result = {
          'rewards': rewards,
          'omegas': omegas}
print(result)

df = pd.DataFrame(result)
df.to_excel('result.xlsx')
