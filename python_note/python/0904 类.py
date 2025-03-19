# 9.4 导入类
from car import Car

my_new_car = Car('audi', 'a4', 2016)
print(my_new_car.get_descriptive_name())

my_new_car.odometer_reading = 23
my_new_car.read_odometer()

# 9.4.2 在一个模块中存储多个类
from electric_car import ElectricCar

my_tesla = ElectricCar('tesla', 'model s', 2016)

print(my_tesla.get_descriptive_name())
my_tesla.battery.describe_battery()
my_tesla.battery.get_range()

# 9.4.3 从一个模块中导入多个类
# from car import Car, ElectricCar
from car import Car
from electric_car import ElectricCar
my_beetle = Car('volkswagen', 'beetle', 2016)
print(my_beetle.get_descriptive_name())
my_tesla = ElectricCar('tesla', 'roadster', 2016)
print(my_tesla.get_descriptive_name())

# 9.4.4 导入整个模块  # 推荐
import car
import electric_car
my_beetle = car.Car('volkswagen', 'beetle', 2016)
print(my_beetle.get_descriptive_name())
my_tesla = electric_car.ElectricCar('tesla', 'roadster', 2016)
print(my_tesla.get_descriptive_name())

# 导入模块中的所有类
# from module_name import *  # 不推荐

# 9.4.6 在一个模块中导入另一个模块

# try 9-10
