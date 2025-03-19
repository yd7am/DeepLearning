# 9.1 创建和使用类
class Dog():
    """一次模拟小狗的简单尝试"""
    def __init__(self, name, age):
        """初始化name和age"""
        self.name = name  # 通过实例访问的变量称为属性
        self.age = age

    def sit(self):
        print(self.name.title() + " is now sitting.")

    def roll_over(self):
        print(self.name.title() + " rolled over!")

# 1.方法__init__()
# 每当创建Dog实例对象时，python会自动运行它
# __函数__()为python默认方法
# 形参self必不可少，必须位于其他形参前面

# Python调用这个__init__() 方法来创建Dog 实例时，将自动传入实参self 。
# 每个与类相关联的方法调用都自动传递实参self ，它是一个指向实例本身的引用，让实例能够访问类中的属性和方法。
# 我们创建Dog实例时，Python将调用Dog类的方法__init__() 。我们将通过实参向Dog() 传递名字和年龄；self 会自动传递，
# 因此我们不需要传递它。每当我们根据Dog 类创建实例时，都只需给最后两个形参(name和age)提供值。

# 9.1.2 根据类创建实例
my_dog = Dog('willie', 6)
print(type(my_dog))
print("My dog's name is " + my_dog.name.title() + ".")  # my_dog.name 访问属性
print("My dog is " + str(my_dog.age) + " years old.")

# 2.调用方法
my_dog.sit()
my_dog.roll_over()

# 3.创建多个实例
my_dog = Dog('willie', 6)
your_dog = Dog('lucy', 3)

print("My dog's name is " + my_dog.name.title() + ".")
print("My dog is " + str(my_dog.age) + " years old.")
my_dog.sit()

print("\nYour dog's name is " + your_dog.name.title() + ".")
print("Your dog is " + str(your_dog.age) + " years old.")
your_dog.sit()

# try 9-1
class Restaurant():
    def __init__(self, name, type):
        self.restaurant_name = name
        self.restaurant_type = type

    def describe_restaurant(self):
        print("The restaurant's name is:" + self.restaurant_name.title() +
              "\nThe restaurant's type is:" + self.restaurant_type.title())

    def open_restaurant(self):
        print("The restaurant is opening!")

my_restaurant = Restaurant('ABC', 'fruit')
print("My_restaurant's name is " + my_restaurant.restaurant_name +
      "\nMy restaurant's type is " + my_restaurant.restaurant_type)
print()
my_restaurant.describe_restaurant()
my_restaurant.open_restaurant()

# 9-2
restaurant1 = Restaurant("A", "AA")
restaurant2 = Restaurant("B", "BB")
restaurant3 = Restaurant("C", "CC")
restaurant1.describe_restaurant()
restaurant2.describe_restaurant()
restaurant3.describe_restaurant()

# 9-3
class User():
    def __init__(self, fname, lname):
        self.first_name = fname
        self.last_name = lname

    def describe_user(self):
        print("The user's name is " + self.first_name +
              " " + self.last_name)

    def greet_user(self):
        self.full_name = self.first_name + " " + self.last_name
        print("Hello " + self.full_name)

user1 = User("A", "BC")
user1.describe_user()
user1.greet_user()