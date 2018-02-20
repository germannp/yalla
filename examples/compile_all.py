"""Compile all examples with `python examples/compile_all.py` to avoid typos."""
import os

examples = [file for file in os.listdir('examples') if file.endswith('.cu')]
for expl in examples:
    os.system('nvcc -std=c++11 -arch=sm_61 ' + 'examples/' + expl + ' -o test')
os.system('rm test')