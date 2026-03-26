import numpy as np;
import matplotlib.pyplot as plt;
import math;

# needed variables
order: int = 10;
t_span: tuple[float, float] = (0.0, 5.0);
h: float = 0.01;

# arrays for computed data
t_steps = int((t_span[1] - t_span[0]) / h);
t = np.zeros(t_steps + 1);
x_arr = np.zeros(t_steps + 1);
p_arr = np.zeros(t_steps + 1);
y1_arr = np.zeros(t_steps + 1);
y2_arr = np.zeros(t_steps + 1);

# initial conditions
x: float = 1.0;
p: float = 0.8;
m: float = 0.1;

# computed
y1: float = x**2;  # x^2
y2: float = x**3;  # x^3

# store initial conditions
t[0] = t_span[0];
x_arr[0] = x;
p_arr[0] = p;
y1_arr[0] = y1;
y2_arr[0] = y2;

# matrix vector setup
# higher derivatives with: A @ state_vector + B @ y_jk
state_vector: np.ndarray = np.array([x, p, y1, y2]);
A: np.ndarray = np.array([[0, 1/m,  0, 0],
                          [4, 0,   -8, 0],
                          [0, 0,    0, 0],
                          [0, 0,    0, 0]]);

B: np.ndarray = np.array([[0,   0],
                          [0,   0],
                          [3/m, 0],
                          [0, 2/m]]);

y_jk: np.ndarray = np.array([p*y2, x*p]);

# ── integration loop ──────────────────────────────────────────────────────────
for i in range(t_steps):

    # derivs[k] = k-th derivative of state_vector [x, p, y1, y2]
    derivs = np.zeros((order + 1, 4));
    derivs[0] = state_vector;  # zeroth derivative = current state

    # build each derivative level
    for k in range(1, order + 1):
        y_jk_k = np.zeros(2);

        # Leibniz sum — pull from previous rows of derivs
        for j in range(k):
            c = math.comb(k - 1, j);
            y_jk_k[0] += c * derivs[j][1] * derivs[k-1-j][3];  # p  * y2
            y_jk_k[1] += c * derivs[j][0] * derivs[k-1-j][1];  # x  * p

        derivs[k] = A @ derivs[k-1] + B @ y_jk_k;

    # taylor sum — advance state_vector by one timestep
    new_state = state_vector.copy();
    for k in range(1, order + 1):
        new_state += derivs[k] * (h**k) / math.factorial(k);

    state_vector = new_state;

    # store trajectory
    t[i+1]      = t[i] + h;
    x_arr[i+1]  = state_vector[0];
    p_arr[i+1]  = state_vector[1];
    y1_arr[i+1] = state_vector[2];
    y2_arr[i+1] = state_vector[3];

# ── plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5));
plt.plot(t, x_arr, 'o-', label=f'x(t)  (h={h}, order={order})');
plt.xlabel('t');
plt.ylabel('x');
plt.title('Position x(t)');
plt.grid(True);
plt.legend();
plt.tight_layout();
plt.show();

plt.figure(figsize=(8, 5));
plt.plot(t, p_arr, 's-', label=f'p(t)  (h={h}, order={order})');
plt.xlabel('t');
plt.ylabel('p');
plt.title('Momentum p(t)');
plt.grid(True);
plt.legend();
plt.tight_layout();
plt.show();

plt.figure(figsize=(8, 5));
plt.plot(t, y1_arr, '^-', label=f'y1(t) = x²  (h={h}, order={order})');
plt.xlabel('t');
plt.ylabel('y1');
plt.title('y1(t) = x²');
plt.grid(True);
plt.legend();
plt.tight_layout();
plt.show();

plt.figure(figsize=(8, 5));
plt.plot(x_arr, p_arr, 'p-', label=f'phase portrait  (h={h}, order={order})');
plt.xlabel('x');
plt.ylabel('p');
plt.title('Phase portrait p vs x');
plt.grid(True);
plt.legend();
plt.tight_layout();
plt.show();