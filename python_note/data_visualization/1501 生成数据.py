import matplotlib.pyplot as plt

# squares = [1, 4, 9, 16, 25]
# input_values = list(range(1,6))
# plt.plot(input_values, squares, linewidth=5)
#
# # 设置图标标题，并给坐标轴加上标签
# plt.title("Square Numbers", fontsize=24)
# plt.xlabel("Value", fontsize=14)
# plt.ylabel("Square of Value", fontsize=14)
# # 设置刻度标记的大小
# plt.tick_params(axis='both', labelsize=14)
# plt.show()

# 15.2.3 scatter()散点图
# x_values = list(range(1, 11))
# y_values = [x**2 for x in x_values]
# # plt.scatter(x_values, y_values, c='red', edgecolor='none', s=40)
# plt.scatter(x_values, y_values, c=(0, 0.99, 0), edgecolor='none', s=40)
# # 设置图表标题并给坐标轴加上标签
# plt.title("Square Numbers", fontsize=24)
# plt.xlabel("Value", fontsize=14)
# plt.ylabel("Square of Value", fontsize=14)
# # 设置刻度标记的大小
# plt.tick_params(axis='both', which='major', labelsize=14)
# # 设置每个坐标轴的取值范围
# plt.axis([0, 11, 0, 110])
# plt.show()

# 15.2.8 颜色映射colormap
# 颜色映射用于突出数据的规律，例如用较浅的颜色来显示较小的值，用较深的颜色来显示较大的值。
# x_values = list(range(1001))
# y_values = [x**2 for x in x_values]
#
# plt.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues,
#             edgecolors='none', s=40)
# plt.tick_params(axis='both', which='major', labelsize=14)
# # plt.show()

# 15.2.9 自动保存图标
# 将plt.show() 替换为 plt.savefig()
# plt.savefig('squares_plot.png', bbox_inches='tight')

# try 15-1
# x_values = list(range(1,5001))
# y_values = [x**3 for x in x_values]
# plt.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues, s=40)
# plt.show()


