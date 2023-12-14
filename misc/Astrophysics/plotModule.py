import matplotlib.pyplot as plt
import numpy as np
import time as time
filename = 'toPlotData/NFW.txt'
file = open(filename, 'r')
datasets = []


errors = []
meshes = []
orders = []
nonZeros = []
dataset = [errors, meshes, orders, nonZeros]
for line in file:
    if line == "\n":
        for i in range(len(dataset)):
            dataset[i] = np.array(dataset[i], dtype=float)
        datasets.append(dataset)
        errors = []
        meshes = []
        orders = []
        nonZeros = []
        dataset = [errors, meshes, orders, nonZeros]
    else:

        lineWithoutAuxSymbols = (line
                                 .replace('[', '')
                                 .replace(']', '')
                                 .replace(':', ''))
        splittedLine = lineWithoutAuxSymbols.split()
        errorIndex = splittedLine.index('error')
        meshIndex = splittedLine.index('mesh')
        ordersIndex = splittedLine.index('orders')
        nonZeroIndex = splittedLine.index('nonZeroAmount')
        errors.append(splittedLine[errorIndex + 1])
        meshes.append(splittedLine[meshIndex + 1: ordersIndex])
        orders.append(splittedLine[ordersIndex + 1: nonZeroIndex])
        # print(splittedLine[ordersIndex + 1: nonZeroIndex])
        nonZeros.append(splittedLine[nonZeroIndex + 1])

for i in range(len(dataset)):
        dataset[i] = np.array(dataset[i], dtype=float)
datasets.append(dataset)
plt.figure()
for i in range(3):
    plt.loglog(np.sum(datasets[i][2], axis=1), datasets[i][0], label=f'Dataset {i+1}')
plt.xlabel('Total Orders')
plt.ylabel('Errors')
plt.title('Errors vs Total Orders')
plt.legend()
plt.grid(True)
plt.show()

for i in range(3):
    plt.loglog(datasets[i][3], datasets[i][0])
plt.show()
# Read data from the file
# with open('[ 0.  1. inf].txt', 'r') as file:
#     lines = file.readlines()
#
# # Extract x and y values from the data, ignoring the rightmost list
# x_values = [float(line.split()[0]) for line in lines]
# y_values = [float(line.split()[1]) for line in lines]
#
# # Plot the data
# plt.loglog(data[:, 0], data[:, 1], marker='o', linestyle='-', color='b')
# plt.title('Your Plot Title')
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')
# plt.grid(True)
# plt.show()