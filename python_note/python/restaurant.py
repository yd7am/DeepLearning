class Restaurant():
    def __init__(self, name, type):
        self.restaurant_name = name
        self.restaurant_type = type
        self.number_served = 0

    def describe_restaurant(self):
        print("The restaurant's name is " + self.restaurant_name +
              "\nThe restaurant's type is " + self.restaurant_type)

    def restaurant_number_print(self):
        print("有" + str(self.number_served) + "人在这家餐厅就餐过")

    def set_number_served(self, num):
        self.number_served = num

    def increment_number_served(self, day_num):
        self.number_served += day_num