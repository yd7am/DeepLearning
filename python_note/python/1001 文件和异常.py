# 10.1 从文件中读取数据
# open('pi_digits.txt') 返回一个表示文件pi_digits.txt 的对象
# 关键字with在不再需要访问文件后将其关闭。在这个程序中，注意到我们调用了open() ，但没有调用close()
# 你也可以调用open() 和close() 来打开和关闭文件
# 但这样做时，如果程序存在bug，导致close() 语句未执行，文件将不会关闭。
# 这看似微不足道，但未妥善地关闭文件可能会导致数据丢失或受损。
with open('pi_digits.txt') as file_object:
    contents = file_object.read()  # 返回一个字符串,相比原始文件多了一个空行
    print(contents.rstrip())  # 删除多出来的空行

# 10.1.2 文件路径
# 相对文件路径
# with open('text_files\filename.txt') as file_object:  # text_files位于文件夹python_work中
# 绝对文件路径
# file_path = '/home/ydam/other_files/text_files/filename.txt'
# file_path = 'C:\Users\ydam\other_files\text_files\filename.txt'
# with open(file_path) as file_object:

# 10.1.3 逐行读取
file_name = 'pi_digits.txt'

with open(file_name) as file_object:
    for line in file_object:
        print(line)  # 原始文件每行末尾有一个换行符，print也会加上一个换行符
        # print(line.rstrip())  # 消除原始文件换行符

# 10.1.4 创建一个包含文件各行内容的列表
with open(file_name) as file_object:
    lines = file_object.readlines()

print(lines)

# 10.1.5 使用文件的内容
file_name = 'pi_digits.txt'
with open(file_name) as file_object:
    lines = file_object.readlines()
pi_string = ''
for line in lines:
    pi_string += line.rstrip()

print(pi_string)
print(len(pi_string))

# 10.1.6 包含一百万位的大型文件
filename = 'pi_million_digits.txt'
with open(filename) as file_object:
    lines = file_object.readlines()

pi_string = ''
for line in lines:
    pi_string += line.rstrip()

print(pi_string[:52] + "...")
print(len(pi_string))

# # 10.1.7 圆周率中包含你的生日吗
# birthday = input("Enter your birthday, in the form mmddyy: ")
# if birthday in pi_string:
#     print("Your birthday appears in the first million digits of pi!")
# else:
#     print("Your birthday does not appear in the first million digits of pi.")

# try 10-1
with open('learning_python.txt') as file_object:
    file_string = file_object.read()
    print(file_string)

with open('learning_python.txt') as file_object:
    for line in file_object:
        print(line.rstrip())

with open('learning_python.txt') as file_object:
    string_lists = file_object.readlines()

for string_line in string_lists:
    print(string_line.rstrip())

# 10-2
print()
with open('learning_python.txt') as file_object:
    for line in file_object:
        line2 = line.replace('Python', 'C')  # replace并没有修改原来的变量内容
        print(line2.rstrip())
