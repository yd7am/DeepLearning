# 3.1 list
bicycles = ['tred', 'cannondale', 'redline', 'specialized']
print(bicycles)
# 3.1.1 访问列表元素
print(bicycles[0].title())
print(bicycles[-1])  # 返回倒数第一个元素
print(bicycles[-2])

# try
names = ['A', 'B', 'C', 'D', 'E']
for i in range(len(names)):
    print(f"第{i+1}个人的名字为: {names[i]}")
for i in range(len(names)):
    print(names[i] + ": Hello")

# 3.2 增、删、改
# 3.2.1 改
motorcycles = ['honda', 'yamaha', 'suzuki']
print(motorcycles)
motorcycles[0] = 'ducati'
print(motorcycles)
# 3.2.2 增
# 1.列表末尾
print(motorcycles)
motorcycles.append('aaa')
print(motorcycles)
# 动态创建列表
motorcycles = []
motorcycles.append('honda')
motorcycles.append('yamaha')
motorcycles.append('suzuki')
print(motorcycles)
# 2.列表中插入元素
motorcycles.insert(0, 'ducati')
print(motorcycles)
# 3.2.3 删
del motorcycles[0]
print(motorcycles)
# 默认删除末尾
motorcycles = ['honda', 'yamaha', 'suzuki']
print(motorcycles)
popped_motorcycle = motorcycles.pop()
print(motorcycles)
print(popped_motorcycle)
# 任意位置
motorcycles = ['honda', 'yamaha', 'suzuki']
first_owned = motorcycles.pop(0)
print('The first motorcycle I owned was a ' + first_owned.title() + '.')
# 根据值删除元素
motorcycles = ['honda', 'yamaha', 'suzuki', 'ducati']
print(motorcycles)
motorcycles.remove('ducati')
print(motorcycles)

# try
invite = ['A', 'B', 'C', 'D', 'E']
nInvite = 'D'
invite.remove(nInvite)
invite.append('F')
for i in range(len(invite)):
    print('Welcome: ' + invite[i] + '!')
print(invite)
invite.insert(0, 'G')
invite.insert(len(invite) // 2, 'H')
invite.append('I')
print(invite)
while True:
    ninvite = invite.pop()
    print(f'Sorry:{ninvite}')
    if len(invite) == 2:
        break
print(f'Welcome:{invite}')
del invite[0]
del invite[0]
print(invite)
