# # 8.3 返回值
# def get_formatted_name(first_name, last_name):
#     """返回完整姓名"""
#     full_name = first_name + ' ' + last_name
#     return full_name.title()
#
# musician = get_formatted_name('jimi', 'hendrix')
# print(musician)

# # 8.3.2 让实参变成可选的
# def get_formatted_name(first_name, last_name, middle_name=''):
#     """返回完整姓名"""
#     if middle_name:
#         full_name = first_name + ' ' + middle_name + ' ' + last_name
#     else:
#         full_name = first_name + ' ' + last_name
#     return full_name.title()
#
# musician = get_formatted_name('john', 'hooker', 'lee')
# print(musician)

# # 8.3.3 返回字典
# def build_person(first_name, last_name):
#     """返回一个字典"""
#     person = {'first': first_name, 'last': last_name}
#     return person
#
# musician = build_person('jimi', 'hendrix')
# print(musician)
#
# # 8.3.4 结合使用函数和while循环
# def get_formatted_name(first_name, last_name):
#     full_name = first_name + ' ' + last_name
#     return full_name.title()
#
# while True:
#     print("\nPlease tell me your name:")
#     print("(enter 'q' at any time to quit)")
#     f_name = input("First name: ")
#     if f_name == 'q':
#         break
#     l_name = input("Last name: ")
#     if l_name == 'q':
#         break
#     formatted_name = get_formatted_name(f_name, l_name)
#     print("\nHello, " + formatted_name + "!")
#
# # 8.4 传递列表
# def greet_users(names):
#     for name in names:
#         msg = "Hello, " + name.title() + "!"
#         print(msg)
#
# usernames = ['hannah', 'ty', 'margot']
# greet_users(usernames)

# 8.4.1 在函数中修改列表
# 在函数中对这个列表所做的任何修改都是永久性的
unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []

while unprinted_designs:
    current_design = unprinted_designs.pop()
    print("Printing model: " + current_design)
    completed_models.append(current_design)

print("\nThe following models have been printed:")
for completed_model in completed_models:
    print(completed_model)

def print_models(unprinted_designs, completed_models):
    while unprinted_designs:
        current_design = unprinted_designs.pop()
        print("Printing model: " + current_design)
        completed_models.append(current_design)

def show_completed_models(completed_models):
    print("\nThe following models have been printed:")
    for completed_model in completed_models:
        print(completed_model)

unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []

# print_models(unprinted_designs, completed_models)
# show_completed_models(completed_models)

# 8.4.2 禁止函数修改列表
# function_name(list_name[:])  # 传递列表的副本
print_models(unprinted_designs[:], completed_models)
print(unprinted_designs)