import pandas as pd
import numpy as np
import os

possible_range = 100
matrix_size = 20
time_slot = 300
batch = 6
time_interval = 0.01
startx = 50

path = "C:\\Users\\THINKPAD\\PycharmProjects\\newpytorch\\final\\preprocess\\dataset"
details = ["\\circle", "\\cross", "\\push"]
kinds = ["\\c0", "\\c30", "\\c-30", "\\s0", "\\s30", "\\s-30", "\\p0", "\\p30", "\\p-30"]
writepath = "C:\\Users\\THINKPAD\\PycharmProjects\\newpytorch\\final\\preprocess\\output\\"




def test(table, index):
    count = 0
    x = 0
    y = 0
    z = 0
    for i in range(possible_range):
        if index + i >= len(table) - 1:
            break
        x = pow(table["Linear Acceleration x (m/s^2)"][index + i], 2)
        y = pow(table["Linear Acceleration y (m/s^2)"][index + i], 2)
        z = pow(table["Linear Acceleration z (m/s^2)"][index + i], 2)

        absolute = pow(x + y + z, 0.5)
        if absolute > 1:
            count += 1
    return count > 50

def is_time_to_break(gyr, index, count, time):
    for i in range(count, len(gyr)):
        if gyr["Time (s)"][i] >= time:
            return i + time_slot > len(gyr)
    return True

if __name__ == '__main__':
    for a in range(len(details)):
        for b in range(a * 3, a * 3 + 3):
            for c in range(3):
                currpath = path + details[a] + kinds[b] + "\\" + str(c)
                gyroscopes = None
                accelerations = None
                for file in os.listdir(currpath):
                    # type = b
                    if file.__contains__("Gyr"):
                        gyroscopes = pd.read_csv(currpath + "\\" + file)
                    if file.__contains__("Lin"):
                        accelerations = pd.read_csv(currpath + "\\" + file)
                        print(file)
                print(gyroscopes)
                print(accelerations)
                k = 0
                tempcount = 0
                gyrcount = 0
                for i in range(len(accelerations)):
                    if tempcount != 0:
                        tempcount -= 1
                        continue
                    else:
                        tempcount = 0
                    x = pow(accelerations["Linear Acceleration x (m/s^2)"][i], 2)
                    y = pow(accelerations["Linear Acceleration y (m/s^2)"][i], 2)
                    z = pow(accelerations["Linear Acceleration z (m/s^2)"][i], 2)
                    absolute = pow(x + y + z, 0.5)
                    # print(absolute)
                    if absolute > 1 and test(accelerations, i):
                        if is_time_to_break(gyroscopes, i, gyrcount, accelerations["Time (s)"][i]):
                            print('break')
                            break
                        else:
                            gyrcount = i
                        #linear matrix
                        matrixlax = np.zeros((1, matrix_size * matrix_size + 1))
                        matrixlay = np.zeros((1, matrix_size * matrix_size + 1))
                        matrixlaz = np.zeros((1, matrix_size * matrix_size + 1))
                        #gyr matrix
                        matrixgyx = np.zeros((1, matrix_size * matrix_size + 1))
                        matrixgyy = np.zeros((1, matrix_size * matrix_size + 1))
                        matrixgyz = np.zeros((1, matrix_size * matrix_size + 1))
                        # 进行数据的smooth,window == 3
                        window_size = 3
                        #la part
                        min_x_la = min(accelerations['Linear Acceleration x (m/s^2)'][:])
                        max_x_la = max(accelerations['Linear Acceleration x (m/s^2)'][:])
                        min_y_la = min(accelerations['Linear Acceleration y (m/s^2)'][:])
                        max_y_la = max(accelerations['Linear Acceleration y (m/s^2)'][:])
                        min_z_la = min(accelerations['Linear Acceleration z (m/s^2)'][:])
                        max_z_la = max(accelerations['Linear Acceleration z (m/s^2)'][:])
                        all_max_la = max(max_x_la, max_y_la, max_z_la)
                        all_min_la = min(min_x_la, min_y_la, min_z_la)
                        bias_la = all_max_la - all_min_la
                        for j in range(window_size, time_slot):
                            former_sum_X = accelerations['Linear Acceleration x (m/s^2)'][j - window_size + 1:j].sum()
                            former_sum_Y = accelerations['Linear Acceleration y (m/s^2)'][j - window_size + 1:j].sum()
                            former_sum_Z = accelerations['Linear Acceleration z (m/s^2)'][j - window_size + 1:j].sum()

                            curr_X = accelerations['Linear Acceleration x (m/s^2)'][j]
                            curr_Y = accelerations['Linear Acceleration y (m/s^2)'][j]
                            curr_Z = accelerations['Linear Acceleration z (m/s^2)'][j]

                            accelerations['Linear Acceleration x (m/s^2)'][j] = (former_sum_X + curr_X) / window_size
                            accelerations['Linear Acceleration y (m/s^2)'][j] = (former_sum_Y + curr_Y) / window_size
                            accelerations['Linear Acceleration z (m/s^2)'][j] = (former_sum_Z + curr_Z) / window_size
                            accelerations['Linear Acceleration z (m/s^2)'][j] = (accelerations[
                                                                                     'Linear Acceleration z (m/s^2)'][
                                                                                     j] - all_min_la) / bias_la
                            accelerations['Linear Acceleration z (m/s^2)'][j] = (accelerations[
                                                                                     'Linear Acceleration z (m/s^2)'][
                                                                                     j] - all_min_la) / bias_la
                            accelerations['Linear Acceleration z (m/s^2)'][j] = (accelerations[
                                                                                     'Linear Acceleration z (m/s^2)'][
                                                                                     j] - all_min_la) / bias_la
                        for j in range(time_slot):
                            matrixlax[0][startx + j] = accelerations["Linear Acceleration x (m/s^2)"][j]
                            matrixlay[0][startx + j] = accelerations["Linear Acceleration y (m/s^2)"][j]
                            matrixlaz[0][startx + j] = accelerations["Linear Acceleration z (m/s^2)"][j]
                        matrixlax[0][400] = b
                        matrixlay[0][400] = b
                        matrixlaz[0][400] = b
                        filenameX = writepath + 'laX.csv'
                        filenameY = writepath + 'laY.csv'
                        filenameZ = writepath + 'laZ.csv'
                        np.savetxt(filenameX, np.asarray(matrixlax), delimiter=",")
                        np.savetxt(filenameY, np.asarray(matrixlay), delimiter=",")
                        np.savetxt(filenameZ, np.asarray(matrixlaz), delimiter=",")

                        #gyr part
                        min_x = min(gyroscopes['Gyroscope x (rad/s)'][:])
                        max_x = max(gyroscopes['Gyroscope x (rad/s)'][:])
                        min_y = min(gyroscopes['Gyroscope y (rad/s)'][:])
                        max_y = max(gyroscopes['Gyroscope y (rad/s)'][:])
                        min_z = min(gyroscopes['Gyroscope z (rad/s)'][:])
                        max_z = max(gyroscopes['Gyroscope z (rad/s)'][:])
                        all_max = max(max_x, max_y, max_z)
                        all_min = min(min_x, min_y, min_z)
                        bias = all_max - all_min
                        for j in range(window_size, time_slot):
                            former_sum_X = gyroscopes['Gyroscope x (rad/s)'][j - window_size + 1:j].sum()
                            former_sum_Y = gyroscopes['Gyroscope x (rad/s)'][j - window_size + 1:j].sum()
                            former_sum_Z = gyroscopes['Gyroscope y (rad/s)'][j - window_size + 1:j].sum()

                            curr_X = gyroscopes['Gyroscope y (rad/s)'][j]
                            curr_Y = gyroscopes['Gyroscope y (rad/s)'][j]
                            curr_Z = gyroscopes['Gyroscope y (rad/s)'][j]

                            gyroscopes['Gyroscope x (rad/s)'][j] = (former_sum_X + curr_X) / window_size
                            gyroscopes['Gyroscope y (rad/s)'][j] = (former_sum_Y + curr_Y) / window_size
                            gyroscopes['Gyroscope z (rad/s)'][j] = (former_sum_Z + curr_Z) / window_size
                            gyroscopes['Gyroscope x (rad/s)'][j] = (gyroscopes['Gyroscope x (rad/s)'][
                                                                        j] - all_min) / bias
                            gyroscopes['Gyroscope y (rad/s)'][j] = (gyroscopes['Gyroscope y (rad/s)'][
                                                                        j] - all_min) / bias
                            gyroscopes['Gyroscope z (rad/s)'][j] = (gyroscopes['Gyroscope z (rad/s)'][
                                                                        j] - all_min) / bias
                        for j in range(time_slot):
                            matrixgyx[0][startx + j] = gyroscopes['Gyroscope x (rad/s)'][j]
                            matrixgyy[0][startx + j] = gyroscopes['Gyroscope y (rad/s)'][j]
                            matrixgyz[0][startx + j] = gyroscopes['Gyroscope z (rad/s)'][j]
                        matrixgyx[0][400] = b
                        matrixgyy[0][400] = b
                        matrixgyz[0][400] = b

                        filenameX = writepath + 'gyX.csv'
                        filenameY = writepath + 'gyY.csv'
                        filenameZ = writepath + 'gyZ.csv'
                        np.savetxt(filenameX, np.asarray(matrixgyx), delimiter=",")
                        np.savetxt(filenameY, np.asarray(matrixgyy), delimiter=",")
                        np.savetxt(filenameZ, np.asarray(matrixgyz), delimiter=",")

                        with open(writepath + 'gyX_train.csv', 'ab') as f:
                            f.write(open(writepath + 'gyX.csv', 'rb').read())
                        with open(writepath + 'gyY_train.csv', 'ab') as f:
                            f.write(open(writepath + 'gyY.csv', 'rb').read())
                        with open(writepath + 'gyZ_train.csv', 'ab') as f:
                            f.write(open(writepath + 'gyZ.csv', 'rb').read())
                        with open(writepath + 'laX_train.csv', 'ab') as f:
                            f.write(open(writepath + 'laX.csv', 'rb').read())
                        with open(writepath + 'laY_train.csv', 'ab') as f:
                            f.write(open(writepath + 'laY.csv', 'rb').read())
                        with open(writepath + 'laZ_train.csv', 'ab') as f:
                            f.write(open(writepath + 'laZ.csv', 'rb').read())
                        # f.write(open(path+"\\" + inputfile, 'rb').read())
                        # filenameX = 'test/laX' + str(curr) + name_series[k] + '_test.csv'
                        # filenameY = 'test/laY' + str(curr) + name_series[k] + '_test.csv'
                        # filenameZ = 'test/laZ' + str(curr) + name_series[k] + '_test.csv'
                        # k += 1
                        # if k == len(name_series):
                        #     k = 0
                        # np.savetxt(filenameX, np.asarray(matrixlax), delimiter=",")
                        # np.savetxt(filenameY, np.asarray(matrixlay), delimiter=",")
                        # np.savetxt(filenameZ, np.asarray(matrixlaz), delimiter=",")
                        # accx.loc[location] = {'X': matrixlax, 'label': [0, 0, 0, 1, 0, 0, 0, 0, 0]}
                        # accy.loc[location] = {'Y': matrixlay, 'label': [0, 0, 0, 1, 0, 0, 0, 0, 0]}
                        # accz.loc[location] = {'Z': matrixlaz, 'label': [0, 0, 0, 1, 0, 0, 0, 0, 0]}
                        tempcount = time_slot
                # print("a pair")