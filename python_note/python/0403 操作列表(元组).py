# 4.5 元组(tuple) 不可变的列表被称为元组
# 4.5.1 定义元组
dimensions = (200, 50)
print(dimensions[0])
print(dimensions[1])
# dimensions[0] = 250  # TypeError: 'tuple' object does not support item assignment
# 4.5.2 遍历元组
for dimension in dimensions:
    print(dimension)
# 4.5.3 修改元组变量
dimensions = (400, 100)
for dimension in dimensions:
    print(dimension)

# try
# 4-13
foods = ('rice', 'noodle', 'c', 'd', 'e')
for food in foods:
    print(food)
foods = ('a', 'b', 'c', 'd', 'e')
for food in foods:
    print(food)
print(foods.__len__())