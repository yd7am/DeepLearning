# 8.6 将函数存储在模块中
# 模块 是扩展名为.py的文件，创建一个pizza.py文件
# 导入模块
import pizza

pizza.make_pizza(16, 'pepperoni')
pizza.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')

# 8.6.2 导入特定的函数
# from module_name import function_0, function_1, function_2
from pizza import make_pizza
make_pizza(13, 'mushrooms', 'green peppers', 'extra cheese')

# 8.6.3 使用as给函数指定别名
# from module_name import function_name as fn
from pizza import make_pizza as mp
mp(16, 'pepperoni')
mp(12, 'mushrooms', 'green peppers', 'extra cheese')

# 8.6.4 使用as给模块指定别名
# import module_name as mn
import pizza as p
p.make_pizza(16, 'pepperoni')
p.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')

# 8.6.5 导入模块中的所有函数
from pizza import *
make_pizza(16, 'pepperoni')
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
# 将模块中的每个函数都复制到这个程序文件中。
# 由于导入了每个函数，可通过名称来调用每个函数，而无需使用句点表示法。
# 然而，使用并非自己编写的大型模块时，最好不要采用这种导入方法：
# 如果模块中有函数的名称与你的项目中使用的名称相同，可能导致意想不到的结果：
# 可能遇到多个名称相同的函数或变量，进而覆盖函数，而不是分别导入所有的函数。

# 最佳的做法是，要么只导入你需要使用的函数，要么导入整个模块并使用句点表示法。
# 这能让代码更清晰，更容易阅读和理解。

# 8.7 函数编写指南
# 给函数指定描述性名称，只使用小写字母和下划线
# 包含阐述性注释""" """
# 形参指定默认值时，等号两边不加空格，关键字实参也是如此

# try 8-16
import pizza
# pizza.make_pizza()
from pizza import make_pizza
# make_pizza()
from pizza import make_pizza as mp
# mp()
import pizza as p
# p.make_pizza()
from pizza import *
# make_pizza()

