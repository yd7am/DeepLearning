# 4.4 使用列表的一部分
# 4.4.1 切片
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print(players[0:3])
print(players[:4])
print(players[2:])
print(players[-3:])

# 4.4.2 遍历切片
print("Here are the first three players on my team:")
for player in players[:3]:
    print(player.title())

# 4.4.3 复制列表
my_foods = ['pizza', 'falafel', 'carrot cake']
friend_foods = my_foods[:]  # 将my_foods 的副本存储到friend_foods
# friend_foods = my_foods  # 将新变量friend_foods 关联到包含 在my_foods 中的列表，因此这两个变量都指向同一个列表。
my_foods.append('cannoli')
friend_foods.append('ice cream')

print("My favorite foods are:")
print(my_foods)
print("\nMy friend's favorite foods are:")
print(friend_foods)

# try
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print("The first three items in the list are:", players[:3])
print("Three items from the middle of the list are:", players[1:4])
print("The last three items in the list are:", players[-3:])

# 4-11
pizzas = ['apple', 'pineapple', 'orange']
friend_pizzas = pizzas[:]
pizzas.append('pear')
friend_pizzas.append('fruits')
print("My favorite pizzas are:\n", pizzas)
print("My friend's favorite pizzas are:\n", friend_pizzas)

for food in my_foods:
    print(food)
for friend_food in friend_foods:
    print(friend_food)