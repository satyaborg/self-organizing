# inheritance test
class Person:
  def __init__(self, **config):
    self.firstname = config.get("fname")
    self.lastname = config.get("lname")
    self.x = "Darn!"

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  def __init__(self, **config):
    super(Student, self).__init__(**config)

config = dict(fname="Satya", lname="borg")
# person = Person(**config)
student = Student(**config)
print(student.x)