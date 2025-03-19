# 5.3 if语句
age = 19
if age >= 18:
    print("You are old enough to vote!")
else:
    print("Sorry, you are too young to vote.")

# 5.3.3 if-elif-else结构
age = 12
if age < 4:
    print("Your admission cost is $0.")
elif age < 18:
    print("Your admission cost is $5.")
else:
    print("Your admission cost is $10.")

if age < 4:
    price = 0
elif age < 18:
    price = 5
elif age < 65:
    price = 10
else:
    price = 5
print("Your admission cost is $" + str(price) + ".")

# try
# 5-3
point = 0
alien_color = 'green'
if alien_color == 'green':
    point = point + 5
print("You get 5 point!")
if alien_color == 'green':
    print('Point add 5')
else:
    print('Point add 10')
if alien_color == 'green':
    print('Point add 5')
if alien_color != 'green':
    print('Point add 10')
# 5-5
if alien_color == 'green':
    print('5 points!')
elif alien_color == 'yellow':
    print('10 points!')
else:
    print('15 points!')