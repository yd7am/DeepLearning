# # 7.1 input()
# message = input("Tell me something, and I will repeat it back to you: ")
# print(message)

# # 7.1.2 int()
# # 使用函数input() 时，Python将用户输入解读为字符串
# age = int(input("How old are you? "))
# print(type(age))

# # 7.1.3 求模运算符
# print(8 % 3)
# number = input("Enter a number, and I'll tell you if it's even or odd: ")
# number = int(number)
# if number % 2 == 0:
#     print("\nThe number " + str(number) + " is even.")
# else:
#     print("\nThe number " + str(number) + " is odd.")

# try
# # 7-1
# car = input("Which car are you want to rent? ")
# print("Let me see if I can find you a " + car.title())

# 7.2 while 循环
current_number = 1
while current_number <= 5:
    print(current_number)
    current_number += 1

# # 7.2.2 让用户选择何时退出
# prompt = "\nTell me something, and I will repeat it back to you: "
# prompt += "\nEnter 'quit' to end the program. "
# message = ""
# while message != 'quit':
#     message = input(prompt)
#     if message != 'quit':
#         print(message)

# # 7.2.3 使用标志
# prompt = "\nTell me something, and I will repeat it back to you: "
# prompt += "\nEnter 'quit' to end the program. "
# active = True
# while active:
#     message = input(prompt)
#     if message == 'quit':
#         active = False
#     else:
#         print(message)

# # 7.2.4 使用break退出循环
# prompt = "\nTell me something, and I will repeat it back to you: "
# prompt += "\nEnter 'quit' to end the program. "
#
# while True:
#     message = input(prompt)
#     if message == 'quit':
#         break
#     else:
#         print(message)

# 7.2.5 在循环中使用continue
current_number = 0
while current_number < 10:
    current_number += 1
    if current_number % 2 == 0:
        continue
    print(current_number)

# try 7-4
# while True:
#     msg = input("请输入pizza配料, 并返回给你: ")
#     if msg == 'quit':
#         break
#     print("Your message is: \n" + msg)

# 7-5
while True:
    age = input("\nPlease input your age: ")
    age = int(age)
    if age < 3:
        outlay = 0
    elif age < 12:
        outlay = 10
    else:
        outlay = 15
    print("You need to pay: " + str(outlay) + "$.\ninput '0' to exit.")
    if age == 0:
        break
