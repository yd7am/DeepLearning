def make_pizza(size, *toppings):
    """打印制作的pizza信息"""
    print("\nMaking a " + str(size) +
          "-inch pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)