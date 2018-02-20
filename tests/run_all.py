"""Run all tests with `python tests/run_all.py`. 

Individual tests can be compiled and run like other models.
"""
import os

tests = [file for file in os.listdir('tests') if file.endswith('.cu')]
for test in tests:
    os.system('nvcc -std=c++11 -arch=sm_61 ' + 'tests/' + test + ' -o test')
    os.system('./test')
os.system('rm test')