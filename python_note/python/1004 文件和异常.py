# 10.4 存储数据
# 10.4.1 json.dump() 和 json.load()
import json
numbers = [2, 3, 5, 7, 11, 13]
filename = 'numbers.json'
with open(filename, 'w') as f_obj:
    json.dump(numbers, f_obj)

# 读
with open(filename) as f_obj:
    numbers = json.load(f_obj)

print(numbers)

# 10.4.2 保存和读取用户生成的数据
# username = input("What's your name? ")
# filename = 'username.json'
# with open(filename, 'w') as f_obj:
#     json.dump(username, f_obj)
#     print("We'll remember you when you come back, " + username + "!")
#
# with open(filename) as f_obj:
#     username = json.load(f_obj)
#     print("Welcome back, " + username + "!")

# 10.4.3 重构
def greet_user():
    """问候用户，并指出其名字"""
    filename = 'username.json'
    try:
        with open(filename) as f_obj:
            username = json.load(f_obj)
    except FileNotFoundError:
        username = input("What is your name? ")
        with open(filename, 'w') as f_obj:
            json.dump(username, f_obj)
            print("We'll remember you when you come back, " + username + "!")
    else:
        print("Welcome back, " + username + "!")

greet_user()

# try 10-11
# filename = 'user_like_num.json'
# while True:
#     try:
#         num = input("Input a number you like: ")
#         num = int(num)
#         with open(filename, 'w') as f_obj:
#             json.dump(num, f_obj)
#     except ValueError:
#         print("请输入数字！")
#     else:
#         break
#
# with open(filename) as f_obj:
#     num = json.load(f_obj)
#     print("I know your favorite number! It's " + str(num))

# 10-12
filename = 'remember_num.json'
try:
    with open(filename) as f_obj:
        num = json.load(f_obj)
        print("Your number is " + str(num))
except FileNotFoundError:
    while True:
        try:
            with open(filename, 'w') as f_obj:
                num = input("Please input your number: ")
                num = int(num)
                json.dump(num, f_obj)
        except ValueError:
            print("请输入数字!")
        else:
            break

