import numpy as np
import matplotlib.pyplot as plt
import math

order = 10;
t_span = (0.0, 10.0);
h = 0.01;

t_steps = int((t_span[1] - t_span[0]) / h);
t = np.zeros(t_steps + 1);

# hsitory arrays
y1_arr = np.zeros(t_steps + 1);
y2_arr = np.zeros(t_steps + 1);
y3_arr = np.zeros(t_steps + 1);
y4_arr = np.zeros(t_steps + 1);
y5_arr = np.zeros(t_steps + 1);
y6_arr = np.zeros(t_steps + 1);
y7_arr = np.zeros(t_steps + 1);
y8_arr = np.zeros(t_steps + 1);

# y1 = x, y2 = y, y3 = vx, y4 = vy
y1 = 1.0;
y2 = 0.0;
y3 = 0.0;
y4 = 1.0;

r = math.sqrt(y1**2 + y2**2);
y6 = r;
y5 = r**3;
y7 = 1.0 / r;
y8 = 1.0 / y5;

# initial state
t[0] = t_span[0];
y1_arr[0] = y1;
y2_arr[0] = y2;
y3_arr[0] = y3;
y4_arr[0] = y4;
y5_arr[0] = y5;
y6_arr[0] = y6;
y7_arr[0] = y7;
y8_arr[0] = y8;

# current full state vector
state_vector = np.array([y1, y2, y3, y4, y5, y6, y7, y8], dtype=float);

# linear
A = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float);

# nonlinear
B = np.array([
    [ 0,  0, 0, 0,  0,  0],
    [ 0,  0, 0, 0,  0,  0],
    [-1,  0, 0, 0,  0,  0],
    [ 0, -1, 0, 0,  0,  0],
    [ 0,  0, 3, 0,  0,  0],
    [ 0,  0, 0, 1,  0,  0],
    [ 0,  0, 0, 0, -1,  0],
    [ 0,  0, 0, 0,  0, -3]
], dtype=float);

# common mmultiplication
S0 = y1*y3 + y2*y4;

# nonlinear vector
y_jk = np.array([
    y1*y8,
    y2*y8,
    y6*S0,
    y7*S0,
    (y7**3)*S0,
    y7*y8*S0
], dtype=float);

print("state_vector =", state_vector);
print("A shape =", A.shape);
print("B shape =", B.shape);
print("y_jk =", y_jk);

# leibnitz rule for multiplication of 2 functions
#
# prod2(derivs, 0, 7, n) means
# n-th derivative of (y1*y8)
# based on leibnitz formula
#
def prod2(derivs, a, b, n):
    s = 0.0;
    for j in range(n + 1):
        s += math.comb(n, j) * derivs[j][a] * derivs[n - j][b];
    return s;


# leibnitz rule for 3 multiplications
# 
# prod3(derivs, 6, 6, 6, j) means
# j-th derivative of (y7*y7*y7) = y7^3
#
def prod3(derivs, a, b, c, n):
    s = 0.0;
    for j in range(n + 1):
        for m in range(n - j + 1):
            l = n - j - m;
            s += (
                math.comb(n, j)
                * math.comb(n - j, m)
                * derivs[j][a]
                * derivs[m][b]
                * derivs[l][c]
            );
    return s;

# derivative for the often used part 
# S = y1*y3 + y2*y4
def S_deriv(derivs, n):
    return prod2(derivs, 0, 2, n) + prod2(derivs, 1, 3, n);

# TODO
# add outer time step loop
# update state with new state 
# add derivatives values to current state and update it 
# increment time
# store new state constants

# working storage for taylor computation
derivs = np.zeros((order + 1, 8));
derivs[0] = state_vector;
#
# derivs matrix for every equation and its derivative
# derivs[k][q] = k-th derivative of variable q at the current time step
#
#  columns:
#   0 -> y1
#   1 -> y2
#   2 -> y3
#   3 -> y4
#   4 -> y5
#   5 -> y6
#   6 -> y7
#   7 -> y8

#
# GENERATING higher derivatives
#
for k in range(1, order + 1):

    n = k - 1;
    y_jk_k = np.zeros(6);

    # higher derivatives based on
    #
    #  y_jk = np.array([
    #    y1*y8,
    #    y2*y8,
    #    y6*S0,
    #    y7*S0,
    #    (y7**3)*S0,
    #    y7*y8*S0
    #  ], dtype=float);
    #

    # (y1*y8)^(n)
    y_jk_k[0] = prod2(derivs, 0, 7, n);

    # (y2*y8)^(n)
    y_jk_k[1] = prod2(derivs, 1, 7, n);

    # (y6*S)^(n)
    for j in range(n + 1):
        y_jk_k[2] += math.comb(n, j) * derivs[j][5] * S_deriv(derivs, n - j);

    # (y7*S)^(n)
    for j in range(n + 1):
        y_jk_k[3] += math.comb(n, j) * derivs[j][6] * S_deriv(derivs, n - j);

    # (y7^3*S)^(n)
    for j in range(n + 1):
        y_jk_k[4] += math.comb(n, j) * prod3(derivs, 6, 6, 6, j) * S_deriv(derivs, n - j);

    # (y7*y8*S)^(n)
    for j in range(n + 1):
        y_jk_k[5] += math.comb(n, j) * prod2(derivs, 6, 7, j) * S_deriv(derivs, n - j);

    derivs[k] = A @ derivs[k - 1] + B @ y_jk_k;

# testing
print("derivs[0] =", derivs[0]);
print("derivs[1] =", derivs[1]);
print("derivs[2] =", derivs[2]);
print("derivs[3] =", derivs[3]);
print("derivs[10] =", derivs[10]);

