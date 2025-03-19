# 6.3 遍历字典
user_0 = {
    'user_name': 'efermi',
    'first': 'enrico',
    'last': 'fermi',
}

for key, value in user_0.items():
    print("\nKey: " + key)
    print("Value: " + value)

favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
}
for name, language in favorite_languages.items():
    print(name.title() + "'s favorite_language is " +
          language.title() + '.')

# 6.3.2 遍历字典中所有键
for name in favorite_languages.keys():
    print(name.title())
# 遍历字典时，会默认遍历所有的键：
for name in favorite_languages:
    print(name.title())

friends = ['phil', 'sarah']
for name in favorite_languages.keys():
    print(name.title())
    if name in friends:
        print("  Hi " + name.title() +
              ", I see your favorite language is " +
              favorite_languages[name].title() + "!")

if 'erin' not in favorite_languages.keys():
    print("Erin, please take our poll!")

# 6.3.3 按顺序遍历字典中所有键
favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
}
for name in sorted(favorite_languages.keys()):
    print(name.title() + ", thank you for taking the poll.")

# 6.3.4 遍历字典所有值
for language in favorite_languages.values():
    print(language.title())
# 避免重复
for language in set(favorite_languages.values()):
    print(language.title())

# try 6-4
dictionarys = {
    'list': '[]',
    'tuple': '()',
    'dict': '{}',
    '.keys()': '返回字典键列表',
    '.items()': '返回字典键值对(key, value)元组列表',
}
for key, value in dictionarys.items():
    print("The " + key + " means: " + value + ".")

dictionarys['.value()'] = '返回字典值列表'
dictionarys['for key in sorted(dict.keys())'] = '排序遍历'
dictionarys['for value in set(dict.values())'] = '避免重复'

for key, value in dictionarys.items():
    print("The " + key + " means: " + value + ".")

# 6-5
rivers = {
    'nile': 'egypt',
    'huanghe': 'china',
    'changjiang': 'china',
}
for river, country in rivers.items():
    print(f"The {river.title()} runs through {country.title()}.")
for river in rivers.keys():
    print(river)
for country in set(rivers.values()):
    print(country)

# 6-6
favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
}
receive_poll = ['sarah', 'phil', 'abc', 'fds', 'jen', 'edward']
for person in receive_poll:
    if person in favorite_languages.keys():
        print("Thanks you for your poll, " + person)
    else:
        print("Please participate in the adjustment, " + person)
