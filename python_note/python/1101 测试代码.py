# 11.1 测试函数
def get_formatted_name(first, last):
    full_name = first + ' ' + last
    return full_name.title()

# 11.1.2 可通过的测试
import unittest

# class NamesTestCase(unittest.TestCase):  # 继承unittest.TestCase类
#     def test_first_last_name(self):  # 所有以test打头的方法都将自动运行
#         formatted_name = get_formatted_name('janis', 'joplin')
#         self.assertEqual(formatted_name, 'Janis Joplin')  # 断言方法
#
# # https://blog.csdn.net/liguofang_527/article/details/105687781
# unittest.main

# 11.1.3 不能通过的测试
# def get_formatted_name(first, middle, last):
#     full_name = first + ' ' + middle + ' ' + last
#     return full_name.title()
#
# class NamesTestCase(unittest.TestCase):  # 继承unittest.TestCase类
#     def test_first_last_name(self):  # 所有以test打头的方法都将自动运行
#         formatted_name = get_formatted_name('janis', 'joplin')
#         self.assertEqual(formatted_name, 'Janis Joplin')  # 断言方法
#
# unittest.main
# TypeError: get_formatted_name() missing 1 required positional argument: 'last'

# 11.1.4 测试未通过怎么办
# def get_formatted_name(first, last, middle=''):
#     if middle:
#         full_name = first + ' ' + middle + ' ' + last
#     else:
#         full_name = first + ' ' + last
#     return full_name.title()
#
# class NamesTestCase(unittest.TestCase):  # 继承unittest.TestCase类
#     def test_first_last_name(self):  # 所有以test打头的方法都将自动运行
#         formatted_name = get_formatted_name('janis', 'joplin')
#         self.assertEqual(formatted_name, 'Janis Joplin')  # 断言方法
#
#
# unittest.main

# 11.1.5 添加新测试
# def get_formatted_name(first, last, middle=''):
#     if middle:
#         full_name = first + ' ' + middle + ' ' + last
#     else:
#         full_name = first + ' ' + last
#     return full_name.title()
#
# class NamesTestCase(unittest.TestCase):  # 继承unittest.TestCase类
#     def test_first_last_name(self):  # 所有以test_打头的方法都将自动运行
#         formatted_name = get_formatted_name('janis', 'joplin')
#         self.assertEqual(formatted_name, 'Janis Joplin')  # 断言方法
#
#     def test_first_last_middle_name(self):  # 方法名必须以test_打头
#         formatted_name = get_formatted_name('wolfgang', 'mozart', 'amadeus')
#         self.assertEqual(formatted_name, 'Wolfgang Amadeus Mozart')
#
#
# unittest.main

# try 11-1
# def get_city_country(city, country):
#     string = city + ', ' + country
#     return string.title()
#
# class NameTestCase(unittest.TestCase):
#     def test_city_country(self):
#         string = get_city_country('santiago', 'chile')
#         self.assertEqual(string, 'Santiago, Chile')
#
#
# if __name__ == '__main__':
#     unittest

