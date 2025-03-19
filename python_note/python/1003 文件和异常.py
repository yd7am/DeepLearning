# 10.3 异常
# try-except
# 10.3.1 处理ZeroDivisionError异常

# print(5/0)  # ZeroDivisionError: division by zero

# 10.3.2 使用try-except代码块
try:
    print(5/0)
except ZeroDivisionError:
    print("You can't divide by zero!")

# 10.3.3 使用异常避免崩溃
# print("Give me two numbers, and I'll divide them.")
# print("Enter 'q' to quit.")
#
# while True:
#     first_number = input("\nFirst number: ")
#     if first_number == 'q':
#         break
#     second_number = input("Second number: ")
#     if second_number == 'q':
#         break
#     answer = int(first_number) / int(second_number)
#     print(answer)
# 程序崩溃可不好，但让用户看到traceback也不是好主意。通过traceback获悉你不希望他知道的信息。
# 例如，他将知 道你的程序文件的名称，还将看到部分不能正确运行的代码。
# 有时候，训练有素的攻击者可根据这些信息判断出可对你的代码发起什么样的攻击。5

# 10.3.4 else代码块
# print("Give me two numbers, and I'll divide them.")
# print("Enter 'q' to quit.")
#
# while True:
#     first_number = input("\nFirst number: ")
#     if first_number == 'q':
#         break
#     second_number = input("Second number: ")
#     if second_number == 'q':
#         break
#     try:
#         answer = int(first_number) / int(second_number)
#     except ZeroDivisionError:
#         print("You can't divide by 0!")
#     else:
#         print(answer)

# try-except-else 代码块的工作原理大致如下：
# Python尝试执行try代码块中的代码；只有可能引发异常的代码才需要放在try语句中。
# 有时候，有一些仅在try代码块成功执行时才需要运行的代码；这些代码应放在else代码块中。
# except代码块告诉Python，如果它尝试运行try代码块中的代码时引发了指定的异常，该怎么办。

# 10.3.5 处理 FileNotFoundError异常
filename = 'alice1.txt'
# with open(filename) as f_obj:
#     contents = f_obj.read()  # FileNotFoundError: [Errno 2] No such file or directory: 'alice.txt'

try:
    with open(filename) as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file " + filename + " does not exist."
    print(msg)

# 10.3.6 分析文本
title = "Alice in Wonderland"
print(title.split())

filename = 'alice.txt'
try:
    with open(filename, encoding='utf-8') as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file " + filename + " does not exist."
    print(msg)
else:
# 计算文件大致包含多少个单词
    words = contents.split()
    num_words = len(words)
    print("The file " + filename + " has about " + str(num_words) + " words.")

# 10.3.7 使用多个文件
def count_words(filename):
    """计算一个文件中大致包含多少个单词"""
    try:
        with open(filename, encoding='utf-8') as f_obj:
            contents = f_obj.read()
    except FileNotFoundError:
        msg = filename + " does not exist."
        print(msg)
    else:
        words = contents.split()
        print(filename + " has about " + str(len(words)) + " words.")

count_words('alice.txt')

# 10.3.8 失败时一声不吭
# pass
# try:
#     --snip--
# except FileNotFoundError:
#     pass
# else:
#     --snip--

# try 10-6\7
# print("(input 'q' to exit)")
# while True:
#     try:
#         first_num = input("Please input the first number: ")
#         if first_num == 'q':
#             break
#         second_num = input("input the second number: ")
#         sum = int(first_num) + int(second_num)
#     except ValueError:
#         print("Error: Please input number!")
#     else:
#         print("The sum is " + str(sum))

# 10-8
def print_file(filename):
    try:
        with open(filename, encoding='utf-8') as f_obj:
            contents = f_obj.read()
    except FileNotFoundError:
        print("文件不存在!")
    else:
        print(contents)

print_file('cats.txt')
print_file('dogs.txt')
print_file('sagsasdf.txt')
