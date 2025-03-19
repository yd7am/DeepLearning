# try 8-9
magicians = ['abc', 'def', 'ghi']
def show_magicians(magicians):
    for magician in magicians:
        print(magician)

show_magicians(magicians)

# 8-10
def make_great(magicians):
    for index in range(len(magicians)):
        magicians[index] = "The Great " + magicians[index]
    return magicians

gm = make_great(magicians[:])
show_magicians(magicians)
show_magicians(gm)

# 8.5 传递任何数量的实参
def make_pizza(*toppings):
    """打印顾客点的所有配料"""
    print(toppings)  # 函数将收到的所有值封装到toppings元组中

make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')

def make_pizza2(*toppings):
    print("\nMaking a pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)

make_pizza2('pepperoni')
make_pizza2('mushrooms', 'green peppers', 'extra cheese')

# 8.5.1 结合使用位置实参和任意数量实参
#  Python 如果要让函数接受不同类型的实参，必须在函数定义中将接纳任意数量实参的形参放在最后。 先匹配位置实参和关键字实参，再将余下的实参都收集
def make_pizza3(size, *toppings):
    print("\nMaking a " + str(size) +
          "-inch pizza witch the following toppings:")
    for topping in toppings:
        print("- " + topping)

make_pizza3(16, 'pepperoni')
make_pizza3(12, 'mushrooms', 'green peppers', 'extra cheese')

# 8.5.2 使用任意数量的关键字实参
# 将函数编写成能够接受任意数量的键值对
def build_profile(first, last, **user_info):
    profile = {}
    profile['first_name'] = first
    profile['last_name'] = last
    for key, value in user_info.items():
        profile[key] = value
    return profile

user_profile = build_profile('albert', 'einstein',
                             location='princeton',
                             field='physics')
print(user_profile)

# try
# 8-12
def add_foods(*elements):
    for element in elements:
        print(element)

add_foods('a', 'b', 'c')

# 8-13
def build_profile2(fist, last, **info):
    dict_info = {}
    dict_info['fist_name'] = fist
    dict_info['last_name'] = last
    for key, value in info.items():
        dict_info[key] = value
    return dict_info

person = build_profile2('a', 'bc', color='black', hobby='game', age=18)
print(person)

# 8-14
def make_car(producer, type, **info):
    car = {}
    car['producer'] = producer
    car['type'] = type
    for key, value in info.items():
        car[key] = value
    return car

car = make_car('subaru', 'outback', color='blue', tow_package=True)
print(car)