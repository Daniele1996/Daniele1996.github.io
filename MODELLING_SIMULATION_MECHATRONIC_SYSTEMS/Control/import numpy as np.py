import numpy as np
import sympy as sym
K = sym.symbols('K')
Kp1 = sym.symbols('Kp1')
Kp2 = sym.symbols('Kp2')
x1d = sym.symbols('x1d')
x2d = sym.symbols('x2d')

M = sym.Matrix([[K+Kp1,-K],[-K,K+Kp2]])
arr = sym.Matrix([Kp1*x1d,Kp2*x2d])
print((M.inv())@arr)