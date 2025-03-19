# 6.4 嵌套
# 6.4.1 字典列表
alien_0 = {'color': 'green', 'points': 5}
alien_1 = {'color': 'yellow', 'points': 10}
alien_2 = {'color': 'red', 'points': 15}

aliens = [alien_0, alien_1, alien_2]

for alien in aliens:
    print(alien)

# 创建一个存储外形人的空list
aliens = []
# 创建30个绿色外星人
for alien_number in range(30):
    new_alien = {'color': 'green', 'points': 5}
    aliens.append(new_alien)
# 显示前五个
for alien in aliens[:5]:
    print(alien)
print("...")
print("Total number of aliens: " + str(len(aliens)))

aliens = []
for num in range(30):
    new_alien = {'color': 'green', 'points': 5, 'speed': 'slow'}
    aliens.append(new_alien)

for alien in aliens[:3]:
    if alien['color'] == 'green':
        alien['color'] = 'yellow'
        alien['speed'] = 'medium'
        alien['points'] = 10

for alien in aliens[:5]:
    print(alien)
print("...")

# 6.4.2 在字典中存储列表
pizza = {
    'crust': 'thick',
    'toppings': ['mushrooms', 'extra cheese'],
}
print("You ordered a " + pizza['crust'] + "-crust pizza " +
      "with the following toppings:")
for topping in pizza['toppings']:
    print("\t" + topping)

favorite_languages = {
    'jen': ['python', 'ruby'],
    'sarah': ['c'],
    'edward': ['ruby', 'go'],
    'phil': ['python', 'haskell'],
}
for name, languages in favorite_languages.items():
    print("\n" + name.title() + "'s favorite languages are:")
    for language in languages:
        print("\t" + language.title())

# 6.4.3 字典中的字典
users = {
    'aeinstein': {
        'first': 'albert',
        'last': 'einstein',
        'location': 'princeton',
    },
    'mcurie': {
        'first': 'marie',
        'last': 'curie',
        'location': 'paris',
    },
}
for user_name, user_info in users.items():
    print('\nUsername: ' + user_name)
    full_name = user_info['first'] + " " + user_info['last']
    location = user_info['location']

    print("\tFull name: " + full_name.title())
    print("\tLocation: " + location.title())

# 6-7
person_0 = {
    'first_name': 'A',
    'last_name': 'bc',
    'age': 18,
    'city': 'Tianjin',
}
person_1 = {
    'first_name': 'W',
    'last_name': 'sf',
    'age': 28,
    'city': 'Beijing',
}
person_2 = {
    'first_name': 'S',
    'last_name': 'f',
    'age': 17,
    'city': 'Tianjin',
}
peoples = [person_0, person_1, person_2]
for person in peoples:
    print(f"My friend's name is {person['first_name'] + person['last_name']}," +
          f" His age is {person['age']}, and he lives in {person['city']}."
          )
# 6-8
pet_0 = {
    'name': 'A',
    'type': 'small',
    'host': 'abc',
}
pet_1 = {
    'name': 'B',
    'type': 'middle',
    'host': 'bcd',
}
pet_3 = {
    'name': 'C',
    'type': 'big',
    'host': 'cde',
}
pets = [pet_0, pet_1, pet_3]
for pet in pets:
    print(pet)

# 6-9
favorite_places = {
    'abc': ['tianjin', 'shenzhen', 'guangzhou'],
    'bcd': ['beijing', 'dongguan'],
    'cdf': ['shanghai'],
}
for person, places in favorite_places.items():
    print("\n" + person + " like the:")
    for place in places:
        print("\t" + place)

# 6-11
cities = {}
cities['tianjin'] = {'country': 'china', 'population': 1000000, 'fact': 'beauty'}
cities['paris'] = {'country': 'france', 'population': 516516, 'fact': 'nice'}
cities['tokyo'] = {'country': 'japan', 'population': 354612, 'fact': 'beauty'}

for city, info in cities.items():
    print("\n" + city + " is in " + info['country'] + ", " +
          "it's population is " + str(info['population']) + ", " +
          "it's fact is " + info['fact'] + '.'
          )

