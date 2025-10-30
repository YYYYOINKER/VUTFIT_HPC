#
#   System for experiment 1
#   Author: Pavol Mihalik, VUT FIT
#
#   ODE system: 
#   y' = -ω z ; y(0) = 0
#   z' = ω y  ; z(0) = 1
#

from numba import njit;
import numpy as np;
import math;
import time;
from numba import njit;


# Function defining the ODE
def f_y(t: float, y: float, z:float, omega: float
          ) -> float:

    # Right hand sides in: y' = -ω z ; y(0) = 0
    return -omega * z;


def f_z(t: float, y:float, z: float, omega: float
        ) -> float:
    
    # Right hand sides in: z' = ω y  ; z(0) = 1
    return omega * y;

# Error functions
def f_vec(t, Y, omega):
    y, z = Y;
    return np.array([-omega*z, omega*y], dtype=float);

def exact_vec(t, omega):
    # t can be scalar or array
    return np.vstack([np.sin(omega*t), np.cos(omega*t)]);  # shape (2, len(t))

def step_euler(t, Y, h, omega):
    return Y + h * f_vec(t, Y, omega);

def step_rk2_heun(t, Y, h, omega):
    k1 = f_vec(t,     Y,           omega);
    k2 = f_vec(t + h, Y + h*k1,    omega);
    return Y + 0.5*h*(k1 + k2);

def step_rk4(t, Y, h, omega):
    k1 = f_vec(t,           Y,               omega);
    k2 = f_vec(t+0.5*h,     Y+0.5*h*k1,      omega);
    k3 = f_vec(t+0.5*h,     Y+0.5*h*k2,      omega);
    k4 = f_vec(t+h,         Y+h*k3,          omega);
    return Y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4);

def step_taylor_vec(t, Y, h, omega, order=10):
    
    y, z = float(Y[0]), float(Y[1]);
    # A^0 = I
    p11 = 1.0; p12 = 0.0;
    p21 = 0.0; p22 = 1.0;
    # start with n=0 term
    y_next = y;
    z_next = z;
    c = 1.0;
    for n in range(1, order+1):
        c *= h / n;  # h^n/n!
        # update A_power = A_power @ A (specialized)
        np11 =  p12 * omega;
        np12 = -p11 * omega;
        np21 =  p22 * omega;
        np22 = -p21 * omega;
        p11, p12, p21, p22 = np11, np12, np21, np22;
        ay = p11 * y + p12 * z;
        az = p21 * y + p22 * z;
        y_next += c * ay;
        z_next += c * az;
    return np.array([y_next, z_next], dtype=float);


# Local eror over time
def lte_over_time(t_span, h, omega, order=10, norm=np.linalg.norm):
    t0, t1 = t_span;
    N = int((t1 - t0)/h);
    t = t0 + np.arange(N)*h;  # step starts

    lte_E  = np.zeros(N);
    lte_R2 = np.zeros(N);
    lte_R4 = np.zeros(N);
    lte_Tm = np.zeros(N);

    for i, ti in enumerate(t):
        Y_i   = exact_vec(ti,   omega)[:,0];      # exact at step start
        Y_ip1 = exact_vec(ti+h, omega)[:,0];      # exact at step end

        Ye  = step_euler(ti,        Y_i, h, omega);
        Yr2 = step_rk2_heun(ti,     Y_i, h, omega);
        Yr4 = step_rk4(ti,          Y_i, h, omega);
        YTm = step_taylor_vec(ti,   Y_i, h, omega, order=order);

        lte_E[i]  = norm(Ye  - Y_ip1, 2);
        lte_R2[i] = norm(Yr2 - Y_ip1, 2);
        lte_R4[i] = norm(Yr4 - Y_ip1, 2);
        lte_Tm[i] = norm(YTm - Y_ip1, 2);

    L = np.stack([lte_E, lte_R2, lte_R4, lte_Tm], axis=0);
    return t, L;


def global_error_over_time(t_span, h, omega, y0=0.0, z0=1.0, norm=np.linalg.norm):

    t0, t1 = t_span;
    N = int((t1 - t0)/h);
    t = t0 + np.arange(N+1)*h;
    Y_exact = exact_vec(t, omega).T;      # shape (N+1, 2)

    Y_e = np.array([y0, z0], float);
    Y_r2 = Y_e.copy();
    Y_r4 = Y_e.copy();

    gE = np.zeros(N+1); gR2 = np.zeros(N+1); gR4 = np.zeros(N+1);

    for i in range(N):

        ti = t[i];
        Y_e  = step_euler(ti,     Y_e,  h, omega);
        Y_r2 = step_rk2_heun(ti,  Y_r2, h, omega);
        Y_r4 = step_rk4(ti,       Y_r4, h, omega);

        gE[i+1]  = norm(Y_e  - Y_exact[i+1], 2);
        gR2[i+1] = norm(Y_r2 - Y_exact[i+1], 2);
        gR4[i+1] = norm(Y_r4 - Y_exact[i+1], 2);

    return t, np.stack([gE, gR2, gR4], axis=0);


# Specialized Solvers for this system
def euler(f_y: callable, f_z: callable, t_span: tuple[float, float], y0: float,  z0: float, 
          h:float, omega : float) -> tuple[np.ndarray, np.ndarray]:

    # Setup
    t_steps:int = int( (t_span[1] - t_span[0]) / h );
    t:np.ndarray = np.zeros(t_steps + 1);
    y:np.ndarray = np.zeros(t_steps + 1);
    z:np.ndarray = np.zeros(t_steps + 1);

    # Initialization
    y[0] = y0;
    z[0] = z0;
    t[0] = t_span[0];

    # Euler loop
    for i in range(t_steps):
        y[i+1] = y[i] + h * f_y(t[i], y[i], z[i], omega);
        z[i+1] = z[i] + h * f_z(t[i], y[i], z[i], omega);
        t[i+1] = t[i] + h;

    return t, y, z;

def rk4(f_y: callable, f_z: callable,
        t_span: tuple[float, float],
        y0: float, z0: float,
        h: float, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Setup
    t_steps = int((t_span[1] - t_span[0]) / h);
    t = np.zeros(t_steps + 1);
    y = np.zeros(t_steps + 1);
    z = np.zeros(t_steps + 1);

    # Initialization
    y[0], z[0] = y0, z0;
    t[0] = t_span[0];

    for i in range(t_steps):

        # Runge-Kutta half-steps

        # k1
        k1_y = f_y(t[i], y[i], z[i], omega);
        k1_z = f_z(t[i], y[i], z[i], omega);

        # k2
        k2_y = f_y(t[i] + h/2, y[i] + h*k1_y/2, z[i] + h*k1_z/2, omega);
        k2_z = f_z(t[i] + h/2, y[i] + h*k1_y/2, z[i] + h*k1_z/2, omega);

        # k3
        k3_y = f_y(t[i] + h/2, y[i] + h*k2_y/2, z[i] + h*k2_z/2, omega);
        k3_z = f_z(t[i] + h/2, y[i] + h*k2_y/2, z[i] + h*k2_z/2, omega);

        # k4
        k4_y = f_y(t[i] + h, y[i] + h*k3_y, z[i] + h*k3_z, omega);
        k4_z = f_z(t[i] + h, y[i] + h*k3_y, z[i] + h*k3_z, omega);

        # Final calculation
        y[i+1] = y[i] + (h/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y);
        z[i+1] = z[i] + (h/6) * (k1_z + 2*k2_z + 2*k3_z + k4_z);
        t[i+1] = t[i] + h;

    return t, y, z;


def taylor_recursive_diff(
    f_y, f_z,
    t_span,
    y0, z0,
    h, omega,
    order=10
):
    """
    Correct Taylor integrator for:
        y' = -ω z
        z' =  ω y
    Uses full Taylor expansion:
        y(t+h) = Σ (h^n / n!) * y^(n)(t)
    """

    t_steps = int((t_span[1] - t_span[0]) / h);
    t = np.zeros(t_steps + 1);
    y = np.zeros(t_steps + 1);
    z = np.zeros(t_steps + 1);

    # Initial values
    y[0], z[0] = y0, z0;
    t[0] = t_span[0];

    for i in range(t_steps):

        # Zeroth derivative (n=0)
        y_next = y[i];
        z_next = z[i];

        # First derivative (n=1)
        dy = -omega * z[i];   # y' = -ω z
        dz =  omega * y[i];   # z' =  ω y
        c = h;                # h^1 / 1!
        y_next += c * dy;
        z_next += c * dz;

        # Higher derivatives n>=2
        for n in range(2, order+1):
            c *= h / n;               # update Taylor coefficient
            dy, dz = -omega * dz, omega * dy;  # recursive derivatives
            y_next += c * dy;
            z_next += c * dz;

        y[i+1] = y_next;
        z[i+1] = z_next;
        t[i+1] = t[i] + h;

    return t, y, z;




def taylor_recursive_diff_matrix(
    f_y: callable, f_z: callable,
    t_span: tuple[float, float],
    y0: float, z0: float,
    h: float, omega: float,
    order: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proper matrix-based Taylor integrator using:
        Y(t+h) = Σ_{n=0}^order (h^n / n!) * A^n * Y(t)
    with A = [[0, -ω],[ω, 0]].
    Builds A^n by repeated multiplication (A_power = A_power @ A).
    No parity trick, no ω² shortcut.
    """

    t_steps = int((t_span[1] - t_span[0]) / h)
    t = np.zeros(t_steps + 1, dtype=float)
    y = np.zeros(t_steps + 1, dtype=float)
    z = np.zeros(t_steps + 1, dtype=float)

    # System matrix
    A = np.array([[0.0, -omega], [omega, 0.0]], dtype=float);

    # Initial value Vector
    Y = np.array([y0, z0], dtype=float)
    y[0], z[0] = y0, z0
    t[0] = t_span[0]

    for i in range(t_steps):

        # Taylor accumulation
        Y_next = np.zeros(2, dtype=float);

        # A^0 term
        A_power = np.eye(2, dtype=float)   # A^0
        c = 1.0

        # Caluclate first no need for division by 1!
        Y_next += c *  (A_power @ Y)

        for n in range(1, order+1):

            # Calculate next Taylor coefficient h/n! ...
            c *= h / n;

            # Calcualte next matrix
            A_power = A_power @ A;

            # Calculate next Values
            Y_next += c * (A_power @ Y);

        Y = Y_next;
        y[i+1], z[i+1] = Y[0], Y[1];
        t[i+1] = t[i] + h;

    return t, y, z ;




@njit(fastmath=True, cache=True)
def matrix_taylor_step_jit(y, z, omega, h, order):
    """
    One time step via true matrix Taylor:
      Y_{next} = sum_{n=0}^order (h^n/n!) * (A^n Y)
    with A = [[0,-ω],[ω,0]], done with raw scalars (no NumPy).

    Inputs:  y,z at current time
    Output:  y_next, z_next
    """

    # A = [[0, -ω], [ω, 0]]
    w = omega

    # A^0 = I
    p11 = 1.0; p12 = 0.0
    p21 = 0.0; p22 = 1.0

    # Start with n = 0 term: I * Y
    y_next = y
    z_next = z

    # Running Taylor coefficient c_n = h^n / n!, start at c0=1, then update
    c = 1.0

    # Loop n = 1..order
    for n in range(1, order + 1):
        # Update coefficient
        c *= h / n

        # Update A_power = A_power @ A, but specialized for A=[[0,-w],[w,0]]
        # New entries (P*A):
        # [[ p12*w,   -p11*w ],
        #  [ p22*w,   -p21*w ]]
        np11 =  p12 * w
        np12 = -p11 * w
        np21 =  p22 * w
        np22 = -p21 * w
        p11, p12, p21, p22 = np11, np12, np21, np22

        # Add c * (A_power @ Y)
        ay = p11 * y + p12 * z
        az = p21 * y + p22 * z
        y_next += c * ay
        z_next += c * az

    return y_next, z_next

@njit(fastmath=True, cache=True)
def taylor_recursive_diff_matrix_jit(t0, t1, y0, z0, h, omega, order):
    """
    Integrate with the JIT-accelerated matrix Taylor step above.
    Returns arrays (t, y, z).
    """
    t_steps = int((t1 - t0) / h)

    t = np.empty(t_steps + 1, dtype=np.float64)
    y = np.empty(t_steps + 1, dtype=np.float64)
    z = np.empty(t_steps + 1, dtype=np.float64)

    t[0] = t0
    y[0] = y0
    z[0] = z0

    ti = t0
    yi = y0
    zi = z0

    for i in range(t_steps):
        yi, zi = matrix_taylor_step_jit(yi, zi, omega, h, order)
        ti += h
        t[i+1] = ti
        y[i+1] = yi
        z[i+1] = zi

    return t, y, z

# End of file