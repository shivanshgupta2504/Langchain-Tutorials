from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None # Default value setting
    email: EmailStr # Built-in validation
    cgpa: float = Field(gt=0, lt=10, default=5, description="Decimal value representing CGPA of the student") # Constraint

new_student = {'name': "Shivansh"}
new_student_1 = {'name': "Shivansh", 'age': '23'} # Automatic Type Coercion
new_student_2 = {'name': "Shivansh", 'email': "ABC@gmail.com"} # Automatic Type Coercion
new_student_3 = {'name': "Shivansh", 'email': "ABC@gmail.com", 'cgpa': 8.93}
student = Student(**new_student_3)
print(dict(student))
print(student.model_dump_json())

