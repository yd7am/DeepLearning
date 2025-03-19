# 8.1 定义函数
def greet_user():
    """显示简单的问候语"""
    print("Hello!")

greet_user()

# 8.1.1 向函数传递信息
def greet_user(username):
    """显示简单的问候语"""
    print("Hello, " + username.title() + "!")

greet_user('jesse')

# try 8-1
def display_message():
    print("I learn definition")

# 8-2
def favorite_book(title):
    print("One of my favorite books is " + title.title())

favorite_book('aaaaa')

# 8.2 传递实参
# 8.2.1 位置实参
def describe_pet(animal_type, pet_name):
    """显示宠物信息"""
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")

describe_pet('hamster', 'harry')

# 8.2.2 关键字实参
describe_pet(animal_type='hamster', pet_name='harry')

# 8.2.3 默认值
def describe_pet(pet_name, animal_type='dog'):
    """显示宠物信息"""
    print("\nI have a " + animal_type + ".")
    print("My " + animal_type + "'s name is " + pet_name.title() + ".")

describe_pet(pet_name='willie')
describe_pet('willie')
describe_pet('willie', 'cat')

# 8.2.4 等效的函数调用
# def describe_pet(pet_name, animal_type='dog'):
# 一条名为Willie的小狗
describe_pet('willie')
describe_pet(pet_name='willie')
# 一只名为Harry的仓鼠
describe_pet('harry', 'hamster')
describe_pet(pet_name='harry', animal_type='hamster')
describe_pet(animal_type='hamster', pet_name='harry')

# 8.2.5 避免实参错误
# try 8-3
def make_shirt(size, content):
    print("The shirt's size is " + size + ", and the content is " + content + ".")
make_shirt('M', 'asdf')
make_shirt(size='M', content='asdf')
# 8-4
def make_shirt(size='big', content='I love Python'):
    print("The shirt's size is " + size + ", and the content is " + content + ".")
make_shirt()
make_shirt('medium')
make_shirt(content='I love C')

# 8-5
def describe_city(city='reykjavik', country='iceland'):
    print(city.title() + " is in " + country.title() + ".")

describe_city()
describe_city(city='beijing', country='china')
describe_city(city='aaaa')