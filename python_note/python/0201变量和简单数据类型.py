import this
# 2.2 变量
message = "Hello Python world!"
print(message)
message = "Hello Python Crash Course world!"
print(message)

# 变量命名
# 2.3 字符串
"This is a string."
'This is also a string.'
print("It's")
# 2.3.1 修改字符串大小写
name = "ada loveLACE"
print(name.title())
print(name.upper())
print(name.lower())
# 2.3.2 拼接字符串
first_name = "ada"
last_name = "lovelace"
full_name = first_name + " " + last_name
print(full_name)
print("Hello, " + full_name.title() + "!")
# 2.3.3 制表符、换行符
print("\tPython")
print("Python1\nPython2")
# 2.3.4 删除空白
favorite_language = 'python   '
print(favorite_language == favorite_language.rstrip())
favorite_language = favorite_language.rstrip()
# favorite_language.lstrip()  # 删除开头空格
# favorite_language.strip()  # 删除两端空格

# try
user_name = "Eric"
print(f"Hello {user_name}, would you like to learn some Python today?")

# 2.4 数字
print(1 / 3, 3 ** 2)
# 2.4.2 浮点数
print(0.1 + 0.2, 3 * 0.1)
# 2.4.3 使用str()避免类型错误
age = 23
# message = "Happy " + age + "rd Birthday!"  # TypeError: can only concatenate str (not "int") to str
message = "Happy " + str(age) +"rd Birthday!"
print(message)
