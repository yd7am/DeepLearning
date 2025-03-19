# 7.3 使用while循环来处理字典和列表
# 7.3.1 在列表之间移动元素
unconfirmed_users = ['alice', 'brian', 'candace']
confirmed_users = []

while unconfirmed_users:
    current_user = unconfirmed_users.pop()
    print("Verifying user: " + current_user.title())
    confirmed_users.append(current_user)

print("\nThe following users have been confirmed:")
for confirmed_user in confirmed_users:
    print(confirmed_user.title())

# 7.3.2 删除包含特定值的所有列表元素
pets = ['dog', 'cat', 'dog', 'goldfish', 'cat', 'rabbit', 'cat']
print(pets)

while 'cat' in pets:
    pets.remove('cat')

print(pets)

# # 7.3.3 使用用户输入来填充字典
# responses = {}
# polling_active = True
#
# while polling_active:
#     name = input("\nWhat is your name? ")
#     response = input("Which mountain would you like to climb someday? ")
#     responses[name] = response
#     repeat = input("Would you like to let another person respond? (yes/ no) ")
#     if repeat == 'no':
#         polling_active = False
#
# print(responses)

# try 7-8
sandwich_orders = ['fish', 'pork', 'fruits']
finished_sandwiches = []
while sandwich_orders:
    sandwich_order = sandwich_orders.pop()
    finished_sandwiches.append(sandwich_order)
    print(f"I made your {sandwich_order} sandwich")
print("Done!")

# 7-9
sandwich_orders = ['fish', 'pork', 'fruits', 'pastrami', 'pastrami', 'pastrami']
print("The pastrami is sold out!")
while 'pastrami' in sandwich_orders:
    sandwich_orders.remove('pastrami')
print(sandwich_orders)

# 7-10
investigation = {}
flag = True
while flag:
    user_name = input("\nPlease input your name: ")
    place = input("Hey " + user_name + " ,If you could vist one place in the world, where would you go? ")
    investigation[user_name] = place
    exit = input("OK, would you want another people to response? (yes/ no): ")
    if exit == 'no':
        flag = False
print(investigation)