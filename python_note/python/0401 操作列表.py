# 4.1 遍历列表
magicians = ['alice', 'david', 'carolina']
for magician in magicians:
    print(magician)
for magician in magicians:
    print(magician.title() + ", that was a great trick!")
print("the last magician is " + magician.title())

# try
pizzas = ['apple', 'pineapple', 'orange']
for pizza in pizzas:
    print(pizza)
    print('I like pepperoni pizza \n')

# 4.3 创建数值列表
# 4.3.1 range()
for value in range(1,5):
    print(value, ' ',end='')
print()
# 数字列表
numbers = list(range(1,6))
print(numbers)
even_numbers = list(range(2,11,2))
print(even_numbers)

squares = []
for value in range(1,11):
    square = value**2  # 临时变量
    squares.append(square)
print(squares)

# 4.3.3 对数字列表执行简单的统计计算
digits = list(range(1,10))
digits.append(0)
print(digits)
print(min(digits))
print(max(digits))
print(sum(digits))

# 4.3.4 列表解析
squares = [value**2 for value in range(1,11)]
print(squares)

# try
for number in range(1,21):
    print(number, ' ', end=" ")
print()
big_num_list = list(range(1,1000001))
# for num in big_num_list:
#     print(num, ' ', end=' ')
# print()
print(min(big_num_list))
print(max(big_num_list))
print(sum(big_num_list))
odd_num = list(range(1,21,2))
print(odd_num)
three_num = [3*i for i in range(1,30//3+1)]
print(three_num)
cubic_num = [i**3 for i in range(1,11)]
print(cubic_num)