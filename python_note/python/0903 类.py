# try 9-4
from restaurant import Restaurant

restaurant = Restaurant('A', 'AA')
restaurant.describe_restaurant()
restaurant.restaurant_number_print()
restaurant.number_served = 100
restaurant.restaurant_number_print()
restaurant.set_number_served(255)
restaurant.restaurant_number_print()
restaurant.increment_number_served(5)
restaurant.restaurant_number_print()

# 9-5
class User():
    def __init__(self):
        self.login_attempts = 0

    def increment_login_attempts(self):
        self.login_attempts += 1

    def reset_login_attempts(self):
        self.login_attempts = 0

    def print_login_attempts(self):
        print("The login_attempts is " + str(self.login_attempts))


user = User()
user.print_login_attempts()
user.increment_login_attempts()
user.increment_login_attempts()
user.print_login_attempts()
user.reset_login_attempts()
user.print_login_attempts()

# 9.3 继承
# 9.3.1 子类的方法__init__()
class Car():
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name

    def read_odometer(self):
        print("This car has " + str(self.odometer_reading) + " miles on it.")

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can't roll back an odometer!")

    def increment_odometer(self, miles):
        self.odometer_reading += miles

class ElectricCar(Car):
    """电动汽车的独特之处"""
    def __init__(self, make, model, year):
        """初始化父类的属性"""
        super().__init__(make, model, year)  # 让python调用ElectricCar的父类的方法__init__()
        # super()帮助将父类和子类联系起来


my_tesla = ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())

# 9.3.3 给子类定义属性和方法
class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery_size = 70

    def describe_battery(self):
        print("This car has a " + str(self.battery_size) + "-kWh battery.")


my_tesla = ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())
my_tesla.describe_battery()

# 9.3.4 重写父类的方法
# 在子类中定义一个与要重写的父类方法同名。

# 9.3.5 将实例用作属性
class Battery():
    def __init__(self, battery_size=70):
        self.battery_size = battery_size

    def describe_battery(self):
        print("This car has a " + str(self.battery_size) + "-kWh battery.")

    def get_range(self):
        if self.battery_size == 70:
            range = 240
        elif self.battery_size == 85:
            range = 270
        message = "This car can go approximately " + str(range)
        message += " miles on a full charge."
        print(message)


class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery = Battery(85)


my_tesla = ElectricCar('tesla', 'model s', 2016)
print(my_tesla.get_descriptive_name())
my_tesla.battery.describe_battery()
print(my_tesla.battery.battery_size)
my_tesla.battery.get_range()

# 9.3.6 模拟实物

# try 9-6
class IceCreamStand(Restaurant):
    def __init__(self, name, type):
        super().__init__(name, type)
        self.flavors = ['酸', '甜', '苦', '辣']

    def print_ice_cream(self):
        print("The icecream is from the " + self.restaurant_type +
              "-" + self.restaurant_name + ".\nHere are the iceCream:")
        print(self.flavors)


my_ice_cream = IceCreamStand('a','bb')
my_ice_cream.print_ice_cream()

# 9-7
class Admin(User):
    def __init__(self):
        super().__init__()
        self.privileges = ['can add post', 'can delete post', 'can ban user']

    def show_privileges(self):
        print("The admin can:")
        for privilege in self.privileges:
            print(privilege)


admin = Admin()
admin.show_privileges()

# 9-8
class Privileges():
    def __init__(self):
        self.privileges = ['can add post', 'can delete post', 'can ban user']

    def show_privileges(self):
        print("The admin can:")
        for privilege in self.privileges:
            print(privilege)

class Admin2(User):
    def __init__(self):
        super().__init__()
        self.privileges = Privileges()


admin2 = Admin2()
admin2.privileges.show_privileges()
