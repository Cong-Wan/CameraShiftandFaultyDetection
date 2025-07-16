from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("file1", ["/home/wilbur/code/project/Camera_ShiftandFault_Detection/version2.0/Main.py"]),
    Extension("file2", ["/home/wilbur/code/project/Camera_ShiftandFault_Detection/version2.0/mClass.py"])
]

setup(
    ext_modules=cythonize(extensions)
)
