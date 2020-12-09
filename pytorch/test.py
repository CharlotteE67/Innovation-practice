import torch
import numpy as np
import pandas as pd
# def compute_error_(b, w, points):
#     totalError = 0
#     for i in range(len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         totalError += (y - (w*x+b))**2
#     return totalError/float(len(points))
#
# def step_gradient(b_current, w_current, points, learningRate):
#     b_gradient = 0
#     w_gradient = 0
#     N = float(len(points))
#     for i in range(len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         b_gradient += -(2/N)*(y - w_current*x - b_current)
#         w_gradient += -(2/N)*x*(y - w_current*x - b_current)
#     new_b = b_current - learningRate*b_gradient
#     new_w = w_current - learningRate*w_gradient
#     return [new_b, new_w]
#
# def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
#     b = starting_b
#     m = starting_m
#     for i in range(num_iterations):
#         b, m = step_gradient(b, m, np.array(points), learningRate)
#     return [b, m]

# def run():
#     points = np.genfromtxt("randomtest.csv", delimiter=",")
#     learningRate = 0.0001
#     init_b = 0
#     init_m = 0
#     num_iterations = 100
#     print("starting gradient descent at b={0},m={1},error={2}".format(init_b, init_m, compute_error_(init_b, init_m, points)))
#     print("running")
#     [b, m] = gradient_descent_runner(points, init_b, init_m, learningRate, num_iterations)
#     print("after {0} interations, b={1}, m={2}, error={3}".format(num_iterations, b, m, compute_error_(b, m, points)))
#
# def generate_random_seed():
#     file = open('randomtest.csv', 'w')
#     xlimit = 40
#     ylimit = 40
#     for i in range(100):
#         file.write(str(xlimit+np.random.normal(10, 5))+","+str(ylimit+np.random.normal(10, 5))+"\n")
#     print("generate successfully")
#
# def get_min_data(accelerations):
#     x_max = accelerations[0][1]
#     x_min = x_max
#     y_max = accelerations[0][2]
#     y_min = y_max
#     z_max = accelerations[0][3]
#     z_min = z_max
#     # for i in range(1, len(accelerations)):
#

def import_data_and_preprocess():
    accelerations = pd.read_csv("Raw Data.csv")
    print(type(accelerations[['X']]))
    # print(accelerations)
    # ax_pre = accelerations[0][1]
    # ay_pre = accelerations[0][2]
    # az_pre = accelerations[0][3]
    # m = 1
    # file.write(str(accelerations[0, 0]) + "," + str(ax_pre) + "," + str(ay_pre) + "," + str(az_pre) + "\n")
    # for i in range(1, len(accelerations)):
    #     accelerations[i][1] = ax_pre + (accelerations[i][1] - accelerations[i - m][1]) / m
    #     accelerations[i][2] = ay_pre + (accelerations[i][2] - accelerations[i - m][2]) / m
    #     accelerations[i][3] = az_pre + (accelerations[i][3] - accelerations[i - m][3]) / m
    #     ax_pre = accelerations[i][1]
    #     ay_pre = accelerations[i][2]
    #     az_pre = accelerations[i][3]
    #     file.write(str(accelerations[i][0]) + "," + str(accelerations[i][1]) + "," + str(accelerations[i][2]) + "," + str(accelerations[i][3]) + "\n")
    #     m += 1
    # print("okay")
    return accelerations

def normalize():
    accelerations = np.genfromtxt("Smooth.csv", delimiter=",")
    print(min(accelerations[:][1]))

if __name__ == '__main__':
    # normalize()
    accelerations = import_data_and_preprocess()
    # generate_random_seed()
    # run()