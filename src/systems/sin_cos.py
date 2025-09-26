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
def euler(f: callable, t_span: tuple[float, float], y0: float,  z0: float, 
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

def rk4(t_span: tuple[float, float], y0: float, z0: float, 
        h: float, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Setup
    t_steps = int((t_span[1] - t_span[0]) / h);
    t = np.zeros(t_steps + 1);
    y = np.zeros(t_steps + 1);
    z = np.zeros(t_steps + 1);

    # Initialization
    y[0], z[0] = y0, z0;
    t[0] = t_span[0];

    # Main loop
    for i in range(t_steps):

        # Runge-Kutta half-steps
        k1_y = h * f_y(t[i], y[i], z[i], omega);
        k1_z = h * f_z(t[i], y[i], z[i], omega);

        k2_y = h * f_y(t[i] + h/2, y[i] + k1_y/2, z[i] + k1_z/2, omega);
        k2_z = h * f_z(t[i] + h/2, y[i] + k1_y/2, z[i] + k1_z/2, omega);

        k3_y = h * f_y(t[i] + h/2, y[i] + k2_y/2, z[i] + k2_z/2, omega);
        k3_z = h * f_z(t[i] + h/2, y[i] + k2_y/2, z[i] + k2_z/2, omega);

        k4_y = h * f_y(t[i] + h, y[i] + k3_y, z[i] + k3_z, omega);
        k4_z = h * f_z(t[i] + h, y[i] + k3_y, z[i] + k3_z, omega);

        # Final calculation
        y[i+1] = y[i] + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6;
        z[i+1] = z[i] + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6;
        t[i+1] = t[i] + h;

    return t, y, z;


# End of file