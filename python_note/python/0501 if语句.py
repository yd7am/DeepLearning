cars = ['audi', 'bmw', 'subaru', 'toyota']

for car in cars:
    if car == "bmw":
        print(car.upper())
    else:
        print(car.title())

# 5.2.3 检查是否不相等
requested_topping = 'mushrooms'

if requested_topping != 'anchovies':
    print("Hold the anchovies")

# 5.2.4 比较数字
age = 18
print(age == 18)

# 5.2.5 检查多个条件
# 1. and
age_0 = 22
age_1 = 18
print(age_0 >= 21 and age_1 >= 21)
age_1 = 22
print(age_0 >= 21 and age_1 >= 21)

# 2. or
age_0 = 22
age_1 = 18
print(age_0 >= 21 or age_1 >= 21)
age_0 = 18
print(age_0 >= 21 or age_1 >= 21)

# 5.2.6 检查特定值是否包含在列表中
requested_toppings = ['mushrooms', 'onions', 'pineapple']
print('mushrooms' in requested_toppings)
print('pepperoni' in requested_toppings)

# 5.2.7 检查特定值是否不包含在列表中
banned_users = ['andrew', 'carolinaa', 'david']
user = 'marie'
if user not in banned_users:
    print(user.title() + ", you can post a response if you wish.")

# 5.2.8 布尔表达式
game_active = True
can_edit = False

# try
print("asdf" != "fda")
print("ABC".lower() == "abc")
print(3 >= 3)
print(2 > 3 and 1 == 2 or 1 == 1)
print(3 in [1, 2, 3])
print(4 not in [1, 2, 3])