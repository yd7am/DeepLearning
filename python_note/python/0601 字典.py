# 6.1 一个简单的字典
alien_0 = {'color': 'green', 'points': 5}
print(alien_0['color'])
print(alien_0['points'])

# 6.2.2 增
print(alien_0)
alien_0['x_position'] = 0
alien_0['y_position'] = 25
print(alien_0)

# 6.2.3 先创建一个空字典
alien_0 = {}
alien_0['color'] = 'green'
alien_0['points'] = 5
print(alien_0)
alien_0['color'] = 'yellow'
print(alien_0)

alien_0 = {'x_pos': 0, 'y_pos': 25, 'speed': 'medium'}
print("Original x-position: " + str(alien_0['x_pos']))
# 向右移动外星人
if alien_0['speed'] == 'slow':
    x_increment = 1
elif alien_0['speed'] == 'medium':
    x_increment = 2
else:
    x_increment = 3
alien_0['x_pos'] = alien_0['x_pos'] + x_increment
print("New x-position: " + str(alien_0['x_pos']))

# 6.2.5 删
alien_0 = {'color': 'green', 'points': 5}
print(alien_0)

del alien_0['points']
print(alien_0)

# try
# 6-1
person = {
    'first_name': 'A',
    'last_name': 'bc',
    'age': 18,
    'city': 'Tianjin',
}
print(f"My friend's name is {person['first_name'] + person['last_name']}," +
      f" His age is {person['age']}, and he lives in {person['city']}."
      )
# 6-2
favorite_numbers = {
    'A': 5,
    'B': 7,
    'C': 1,
    'D': 3,
    'E': 6,
}