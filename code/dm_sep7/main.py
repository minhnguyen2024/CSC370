# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib as plt
class Student:
    def __init__(self, name="NAME", box_number=-1):
        self.name = name
        self.box_number = box_number

    def get_name(self):
        return self.name

    def get_box_number(self):
        return self.box_number

    def wrap(self, wrap_string):
        return wrap_string + str(self) + wrap_string

    def __str__(self):
        return self.name + " " + str(self.box_number)

def main():
    s1 = Student()
    s2 = Student("Minh", 2)


    # print(s1.get_name())
    # print(s2.get_name())
    # print(s3.get_name())


main()

