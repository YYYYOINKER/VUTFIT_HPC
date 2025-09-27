#
#   System for experiment 1
#   Author: Pavol Mihalik, VUT FIT
#
#   ODE system: 
#   y' = -ω z ; y(0) = 0
#   z' = ω y  ; z(0) = 1
#

import numpy as np;

# Function defining the ODE
def f_y(t: float, y: float, z:float, omega: float
          ) -> float:

    # Right hand sides in: y' = -ω z ; y(0) = 0
    return -omega * z;


def f_z(t: float, y:float, z: float, omega: float
        ) -> float:
    
    # Right hand sides in: z' = ω y  ; z(0) = 1
    return omega * y;


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
    f_y: callable, f_z: callable,
    t_span: tuple[float, float],
    y0: float,
    z0: float,
    h: float,
    omega: float,
    order: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    t_steps = int((t_span[1] - t_span[0]) / h)
    t = np.zeros(t_steps + 1)
    y = np.zeros(t_steps + 1)
    z = np.zeros(t_steps + 1)

    y[0], z[0] = y0, z0
    t[0] = t_span[0]

    for i in range(t_steps):
        # Start with zeroth order (just the current values)
        y_next = y[i]
        z_next = z[i]

        # First derivatives
        dy = -omega * z[i]
        dz =  omega * y[i]

        # First order term
        y_next += h * dy
        z_next += h * dz

        # Higher orders
        for j in range(2, order + 1):
            # Compute j-th derivatives from previous ones
            ddy = -omega * dz
            ddz =  omega * dy
            dy, dz = ddy, ddz

            # Add contribution with division by j
            y_next += (h / j) * dy
            z_next += (h / j) * dz

        # Update arrays
        y[i+1], z[i+1] = y_next, z_next
        t[i+1] = t[i] + h

    return t, y, z


# End of file