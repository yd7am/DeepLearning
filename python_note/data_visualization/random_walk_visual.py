import  matplotlib.pyplot as plt
from random_walk import RandomWalk

# 创建一个RandomWalk实例，并将其包含的点都绘制出来
# rw = RandomWalk()
# rw.fill_walk()
#
# point_numbers = list(range(rw.num_points))
# plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
#             edgecolors='none', s=15)
# # 突出起点和终点
# plt.scatter(rw.x_values[0], rw.y_values[0], c='green', edgecolors='none',
#             s=100)
# plt.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none',
#             s=100)
# # 隐藏坐标轴
# # plt.axes().get_xaxis().set_visible(False)
# # plt.axes().get_yaxis().set_visible(False)
#
# plt.show()

# 15.3.9 增加点数
# rw = RandomWalk(50000)
# rw.fill_walk()
#
# # 设置绘图窗口的尺寸
# plt.figure(figsize=(10, 6))  # 单位英寸
# point_numbers = list(range(rw.num_points))
# plt.scatter(rw.x_values, rw.y_values, c=point_numbers,
#             cmap=plt.cm.Blues, edgecolors='none', s=1)
# # 突出起点和终点
# plt.scatter(rw.x_values[0], rw.y_values[0], c='green', edgecolors='none',
#             s=100)
# plt.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none',
#             s=100)
# plt.show()

# try 15-3
rw = RandomWalk(5000)
rw.fill_walk()
plt.plot(rw.x_values, rw.y_values, linewidth=1)
plt.show()