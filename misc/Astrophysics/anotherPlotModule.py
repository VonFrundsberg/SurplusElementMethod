import matplotlib.pyplot as plt
import numpy as np
import scienceplots
filename = 'toPlotData/easyOnes.txt'
file = open(filename, 'r')
datasets = []

"""READ FROM FILE"""
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
        # meshIndex = splittedLine.index('mesh')
        ordersIndex = splittedLine.index('orders')
        nonZeroIndex = splittedLine.index('nonZeroAmount')
        errors.append(splittedLine[errorIndex + 1])
        # meshes.append(splittedLine[meshIndex + 1: ordersIndex])
        orders.append(splittedLine[ordersIndex + 1: nonZeroIndex])
        # print(splittedLine[ordersIndex + 1: nonZeroIndex])
        nonZeros.append(splittedLine[nonZeroIndex + 1])

for i in range(len(dataset)):
        dataset[i] = np.array(dataset[i], dtype=float)
datasets.append(dataset)
"""PLOT """
# Use the science style
plt.style.use(['science','ieee', 'grid'])
plt.rcParams.update({'figure.dpi': '300'})
line_styles = ['-', '--', ':', '-.']
# labels = ['$v^{optimized \, p}_{BPL}$', '$v^{fixed \, p}_{BPL}$', '$v^3_{BPL}$', '$v^6_{BPL}$']
labels = ['$v^{kernel}_{BPL}$', '$v^{outer}_{BPL}$', '$v^{shell}_{BPL}$']
for i in range(3):
    orders_sum = np.sum(datasets[i][2], axis=1)
    # orders_sum = datasets[i][3]
    # if i == 0:
    #     orders_sum = datasets[i][2][:, 0]
    # if i == 1:
    #     orders_sum = datasets[i][2][:, -1]
    # if i == 2:
    #     orders_sum = datasets[i][2][:, -4]

    errors = datasets[i][0]

        # Use different line styles for each dataset

    plt.loglog(orders_sum, errors, label=labels[i], linestyle=line_styles[i])

    plt.xlabel('Approximation Order')
    plt.ylabel('Error')
    # plt.xlabel('Non-zero elements')
    # plt.title('Errors vs Total Orders')
    # tick_positions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # tick_labels = np.array(np.log2(tick_positions), dtype=int)
    # tick_positions = [1e+1, 1e+2, 1e+3, 1e+4, 1e+5]
    # tick_labels = np.array(np.log10(tick_positions), dtype=int)
    # plt.xticks(tick_positions, [f'$10^{{{pos}}}$' for pos in tick_labels])
    # plt.tick_params(axis='y', which='both', labelright=True, labelleft=False)
    # tick_positions = [2, 3, 20, 30]
    # plt.xticks(tick_positions, [f'{pos}' for pos in tick_positions])

    # tick_positions = [1e+1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15, 1e-17]
    # tick_labels = np.array(np.log10(tick_positions), dtype=int)
    # plt.yticks(tick_positions, [f'$10^{{{pos}}}$' for pos in tick_labels])
    plt.legend()

    # Customize grid appearance

    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

plt.show()

# for i in range(3):
#     orders_sum = datasets[i][3]
#     errors = datasets[i][0]
#
#         # Use different line styles for each dataset
#
#     plt.loglog(orders_sum, errors, label=labels[i], linestyle=line_styles[i])
#
#     plt.xlabel('Non-zero elements')
#     plt.ylabel('Error')
#     # plt.title('Errors vs Total Orders')
#     # tick_positions = [4, 8, 16, 32, 40]
#     # plt.xticks(tick_positions, [f'{pos}' for pos in tick_positions])
#     plt.tick_params(axis='y', which='both', labelright=True, labelleft=False)
#     plt.legend()

    # Customize grid appearance

    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# plt.show()