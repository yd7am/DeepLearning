# 10.2 写入文件
# 10.2.1 写入空文件
filename = 'programming.txt'

# 默认只读,读取模式('r'),写入模式('w'),附加模式('a'),读写('r+')
with open(filename, 'w') as file_object:
    file_object.write("I love programming.")

# 10.2.2 写入多行
with open(filename, 'w') as file_object:
    file_object.write("I love programming.\n")
    file_object.write("I love creating new games.\n")

# 10.2.3 附加到文件
with open(filename, 'a') as file_object:
    file_object.write("I also love finding meaning in large datasets.\n")
    file_object.write("I love creating apps that can run in a browser.\n")

# try 10-3
# name = input("input your name: ")
# with open('guest.txt', 'a') as file_object:
#     file_object.write(name + '\n')

# 10-4
while 1:
    name = input("Please input your name (input 'q' to quit):  ")
    if name == 'q':
        break
    else:
        with open('guest_book.txt', 'a') as file_object:
            file_object.write(name + '\n')
