import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt


df = pd.read_csv('Raw Data.csv')
df_process = df.loc[df["Time"]>11]
df_process = df_process.loc[df_process["Time"]<=14]
time_interval = 0.01

features = ['X','Y','Z']
time_list = np.arange(0,len(df_process)*time_interval,time_interval)
# print(time_list)
color = ['r',"g",'b']

for i in range(len(features)):
    plt.plot(time_list,df_process[features[i]],color = color[i],label = features[i])
    plt.legend(loc = 'upper right')
# plt.savefig("raw.jpg")
plt.show()


# 重新设置index
df_process = df_process.reset_index(drop = True)


# 进行数据的smooth,window == 2
window_size = 3
# print(df_process)
# print(min(df_process['X'][:]))
for i in range(window_size,len(df_process)):
    former_sum_X = df_process['X'][i - window_size + 1:i].sum()
    former_sum_Y = df_process['Y'][i - window_size + 1:i].sum()
    former_sum_Z = df_process['Z'][i - window_size + 1:i].sum()

    curr_X = df_process['X'][i]
    curr_Y = df_process['Y'][i]
    curr_Z = df_process['Z'][i]

    df_process['X'][i] = (former_sum_X+curr_X) / window_size
    df_process['Y'][i] = (former_sum_Y+curr_Y) / window_size
    df_process['Z'][i] = (former_sum_Z+curr_Z) / window_size

for i in range(len(features)):
    plt.plot(time_list,df_process[features[i]],color = color[i],label = features[i])
    plt.legend(loc = 'upper right')
# plt.savefig("smooth.jpg")
plt.show()


#continue to normalize
min_x = min(df_process['X'][:])
max_x = max(df_process['X'][:])
min_y = min(df_process['Y'][:])
max_y = max(df_process['Y'][:])
min_z = min(df_process['Z'][:])
max_z = max(df_process['Z'][:])
all_max = max(max_x, max_y, max_z)
all_min = min(min_x, min_y, min_z)
bias = all_max - all_min
for i in range(len(df_process)):
    df_process['X'][i] = (df_process['X'][i] - all_min) / bias
    df_process['Y'][i] = (df_process['Y'][i] - all_min) / bias
    df_process['Z'][i] = (df_process['Z'][i] - all_min) / bias
for i in range(len(features)):
    plt.plot(time_list,df_process[features[i]],color = color[i],label = features[i])
    plt.legend(loc = 'upper right')
plt.savefig("all_normalize.jpg")
plt.show()

#xyz normalize
# x_bias = max_x - min_x
# y_bias = max_y - min_y
# z_bias = max_z - min_z
# for i in range(len(df_process)):
#     df_process['X'][i] = (df_process['X'][i] - min_x) / x_bias
#     df_process['Y'][i] = (df_process['Y'][i] - min_y) / y_bias
#     df_process['Z'][i] = (df_process['Z'][i] - min_z) / z_bias
#
# for i in range(len(features)):
#     plt.plot(time_list,df_process[features[i]],color = color[i],label = features[i])
#     plt.legend(loc = 'upper right')
# plt.savefig("normalize.jpg")
# plt.show()