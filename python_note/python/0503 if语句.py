# 5.4 使用if处理列表
# 5-8
user_names = ['user_1', 'admin', 'user_2', 'user_3', 'user_4']
for user_name in user_names:
    if user_name == 'admin':
        print("Hello admin, would you like to see a status report?")
    else:
        print(f"Hello {user_name}, thank you for logging in again.")
# 5-9
user_names = []
if user_names:
    print("Hello!")
else:
    print("We need to find some users!")

# 5-10
current_users = ['user_1', 'admin', 'user_2', 'user_3', 'user_4']
new_users = ['admin', 'USER_3', 'user_5', 'user_6', 'user_7']
for new_user in new_users:
    if new_user.lower() in current_users:
        print(f"{new_user} has been used! Please input anothor name")
    else:
        current_users.append(new_user.lower())
        print("done!")
print("Current User: ", current_users)

# 5-11
numbers = list(range(1,10))
number_str = []
for number in numbers:
    if number == 1:
        number_str.append(str(number) + 'st')
    elif number == 2:
        number_str.append(str(number) + 'nd')
    elif number == 3:
        number_str.append('3rd')
    else:
        number_str.append(str(number) + 'th')
print(number_str)
for i in range(9):
    print(number_str[i])
