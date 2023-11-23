import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('toPlotData/[ 0.  1. inf].txt')


# Read data from the file
# with open('[ 0.  1. inf].txt', 'r') as file:
#     lines = file.readlines()
#
# # Extract x and y values from the data, ignoring the rightmost list
# x_values = [float(line.split()[0]) for line in lines]
# y_values = [float(line.split()[1]) for line in lines]
#
# # Plot the data
plt.loglog(data[:, 0], data[:, 1], marker='o', linestyle='-', color='b')
plt.title('Your Plot Title')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.grid(True)
plt.show()