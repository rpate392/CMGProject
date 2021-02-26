import sympy as sym
import numpy as np
from math import pi
from scipy import linalg
v1, v2, v3, q1, q2, tau2, tau3 = sym.symbols('v_1, v_2, v_3, q_1, q_2, tau_2, tau_3')

v1e = 0
v2e = 0
v3e = (250 * 2 * np.pi / 60)
q1e = pi *.5
q2e = 0
tau2e = 0
tau3e = 0

v1dot = -5 * ((200 * tau3 * sym.sin(q2)) + (sym.sin(2 * q2) * v1 * v2) + (2 * sym.cos(q2) * v2 * v3e))/((10 * (sym.sin(q2))**2) - 511)
v2dot = 10 * ((100 * tau2) - (sym.cos(q2) * v1 * v3e))/11
v3dot = -((51100 * tau3) + (5 * sym.sin(2 * q2) * v2 * v3e) + (511 * sym.cos(q2) * v1 * v2))/((10 * (sym.sin(q2)) ** 2) - 511)

f = sym.Matrix([[v1],[v2],[v1dot],[v2dot]])
print(f)


f_num = sym.lambdify((q1, q2, v1, v2), f)


# Create lambda function
A_num = sym.lambdify((q1, q2, v1, v2, tau2, tau3), f.jacobian([q1, q2, v1, v2]))

# Evaluate lambda function at equilibrium point
A = A_num(q1e, q2e, v1e, v2e, tau2e, tau3e)
# Suppress the use of scientific notation when printing small numbers
np.set_printoptions(suppress=True)

# Create lambda functions
B_num = sym.lambdify((q1, q2, v1, v2, tau2, tau3), f.jacobian([tau2, tau3]))

# Evaluate lambda function at equilibrium point
B = B_num(q1e, q2e, v1e, v2e, tau2e, tau3e)


k = np.array([[24.7, 36., 49.7, 15.1.], [0., 0., 0., 0.]])
F = A - (B @ k)

print((A))
print((B))
print((k))
print(F)
A = A.astype(float)
B = B.astype(float)
s = linalg.eigvals(F)

print(s.real)