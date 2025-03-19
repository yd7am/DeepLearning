# 11.2 测试类
# 11.2.2 一个要测试的类
import unittest

class AnonymousSurvey():
    def __init__(self, question):
        self.question = question
        self.responses = []

    def show_question(self):
        print(self.question)

    def store_response(self, new_response):
        self.responses.append(new_response)

    def show_results(self):
        print("survey results:")
        for response in self.responses:
            print("- " + response)

# question = "What language did you first learn to speak?"
# my_survey = AnonymousSurvey(question)
#
# my_survey.show_question()
# print("Enter 'q' at any time to quit.\n")
# while True:
#     response = input("Language: ")
#     if response == 'q':
#         break
#     my_survey.store_response(response)
#
# print("\nThank you to everyone who participated in the survey!")
# my_survey.show_results()

# 11.2.3 测试 AnonymousSurvey 类
# class TestAnonymousSurvey(unittest.TestCase):
#     def test_store_single_response(self):
#         question = "What language did you first learn to speak?"
#         my_survey = AnonymousSurvey(question)
#         my_survey.store_response('English')
#
#         self.assertIn('English', my_survey.responses)
#
#     def test_store_three_responses(self):
#         question = "What language did you first learn to speak?"
#         my_survey = AnonymousSurvey(question)
#         responses = ['English', 'Spanish', 'Mandarin']
#         for response in responses:
#             my_survey.store_response(response)
#
#         for response in responses:
#             self.assertIn(response, my_survey.responses)
#
#
# unittest.main

# 11.2.4 方法setUp() 只需创建对象一次
# class TestAnonymousSurvey(unittest.TestCase):
#
#     def setUp(self):
#         question = "What language did you first learn to speak?"
#         self.my_survey = AnonymousSurvey(question)
#         self.responses = ['English', 'Spanish', 'Mandarin']
#
#     def test_store_single_responses(self):
#         self.my_survey.store_response(self.responses[0])
#         self.assertIn(self.responses[0], self.my_survey.responses)
#
#     def test_store_three_responses(self):
#         for response in self.responses:
#             self.my_survey.store_response(response)
#         for response in self.responses:
#             self.assertIn(response, self.my_survey.responses)
#
#
# unittest.main
