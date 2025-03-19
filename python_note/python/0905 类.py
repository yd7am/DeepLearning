# 9.5 Python标准库
from collections import OrderedDict

favorite_languages = OrderedDict()

favorite_languages['jen'] = 'python'
favorite_languages['sarah'] = 'c'
favorite_languages['edward'] = 'ruby'
favorite_languages['phil'] = 'python'

for name, language in favorite_languages.items():
    print(name.title() + "'s favorite language is " +
          language.title() + ".")

# try
# 9-14
from random import randint

class Die():
    def __init__(self, sides=6):
        self.sides = sides

    def roll_die(self):
        print("-" + str(randint(1, self.sides)) + "-", end="")

sites = [6, 10, 20]
for site in sites:
    die = Die(site)
    for i in range(10):
        die.roll_die()
    print()

# 9.6 类编码风格
# 类名应采用驼峰命名法，即将类名中的每个单词的首字母都大写，而不使用下划线。
# 实例名和模块名都采用小写格式，并在单词之间加上下划线。
