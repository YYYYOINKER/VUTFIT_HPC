#
#   System for experiment 3
#   Author: Pavol Mihalik, VUT FIT
#
#   The kepler problem (n body, n=2)
#   Hamiltonian formalisation
#   
#   H = T + V
#   H(x; y; px; py) = 1/2 (px^2 + py^2) - mu/sqrt(x^2 + y^2)
#
#   deriving ∂H/∂px, ∂H/∂py
#   ∂H/∂px = 1/2 (2px + 0) - 0 = px ; ∂H/∂py = py
#
#   deriving ∂H/∂x, ∂H/∂y
#   ∂H/∂x = 1/2 (0 + 0) - mu * ( -1/2 [x^2 = y^2]^{-3/2} * 2x )
#   - cancel out 1/2 and 2 and minuses, derive wia power rule
#   = mu * x / (x^2 + y ^2)^{3/2}
#   
#   let r = sqrt(x^2 + y^2) we get:
#   ∂H/∂x = mu*x/r^3
#   ∂H/∂y = mu*y/r^3

import numpy as np
import matplotlib.pyplot as plt


# state vector z = [ x; y; px; py ]
# symplectic structure matrix [ [0 ,  0, 1, 0],
#                               [0 ,  0, 0, 1],
#                               [-1,  0, 0, 0],
#                               [0 , -1, 0, 0] ]
#
def make_J(n: int):

    I = np.eye(n)
    #   I = [ [1, 0],
    #         [0, 1] ]
    Z = np.zeros((n, n))
    #   Z = [ [0, 0],
    #         [0, 0] ]
    # return as lego blocks up to desired n
    #               [ [0,  0, 1, 0].
    #   Z  I  =>      [0,  0, 0, 1],
    #   -I Z  =>      [-1, 0, 0, 0],
    #                 [0, -1, 0, 0] ]
    return np.block([[Z, I], [-I, Z]])

# gradient function
def grad_H(z: np.ndarray, mu: float):

    x, y, px, py = z;
    r = np.sqrt(x**2 + y**2);
    return np.array([
        mu * x / r**3,     # ∂H/∂x
        mu * y / r**3,     # ∂H/∂y
        px,                # ∂H/∂px
        py                 # ∂H/∂py
    ]);


# hamiltonian function zdot = J ∇H(z)
def f(z, mu):
    J = make_J(2); # 2D sys.
    return J @ grad_H(z, mu);


def explicit_euler(z0: np.ndarray, t_span: tuple[float, float], h: float, mu: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Explicit Euler integrator for the Kepler problem (Hamiltonian form)
    z' = J ∇H(z)
    """
    
    # Setup
    t_steps = int( (t_span[1] - t_span[0]) / h);
    t = np.zeros(t_steps + 1);
    b = len(z0);
    z = np.zeros( (t_steps + 1, b) );

    # Initial condition
    t[0] = t_span[0];
    z[0] = z0;

    # Euler loop
    for i in range(t_steps):

        z[i+1] = z[i] + h * f(z[i], mu);
        t[i+1] = t[i] + h;

    return t, z;


def symplectic_euler(z0: np.ndarray, t_span: tuple[float, float], h: float, mu: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Symplectic Euler integrator (momentum-first) for the Kepler problem
    """

    # Setup
    t_steps = int( (t_span[1] - t_span[0]) / h);
    t = np.zeros(t_steps + 1);
    b = len(z0);
    z = np.zeros( (t_steps + 1, b) );

    # Initial conditions
    t[0] = t_span[0];
    z[0] = z0;

    # symplectic loop momentum first
    for i in range(t_steps):

        x, y, px, py = z[i];
        r = np.sqrt(x**2 + y**2);

        # Momentum first
        px_new = px - h * mu * x / r ** 3;
        py_new = py - h * mu * y / r ** 3;

        # Position second
        x_new = x + h * px_new;
        y_new = y + h * py_new;

        # Update Zn+1
        z[i+1] = np.array([x_new, y_new, px_new, py_new]);
        t[i+1] = t[i] + h;
    
    return t, z;


# Energy function
def H(z: np.ndarray, mu: float):

    x, y, px, py = z;
    r = np.sqrt(x**2 + y**2);
    return 0.5 * (px**2 + py**2) - mu / r;


# Action principle function
# S≈∑​(px,i​Δxi​+py,i​Δyi​−Hi​Δt)
def action(t: np.ndarray, z: np.ndarray, mu: float):

    """
    Compute the discrete action integral:
    S = ∫ (p·dx - H dt)
    Approximated by summation over steps.
    """
    h = t[1] - t[0];
    S = 0.0;

    # Main approx loop
    for i in range(len(t) -1):

        x, y, px, py = z[i];
        x_next, y_next, *_ = z[i+1];

        dx = x_next - x;
        dy = y_next - y;

        H_i = H(z[i], mu);

        # p·dx - Hdt
        S += (px * dx + py * dy - H_i * h);

    return S;


def cumulative_action(t, z, mu):

    h = t[1] - t[0]
    S = np.zeros(len(t))

    for i in range(1, len(t)):

        x, y, px, py = z[i-1]
        x_next, y_next, *_ = z[i]

        dx = x_next - x
        dy = y_next - y

        H_i = H(z[i-1], mu)
        S[i] = S[i-1] + (px * dx + py * dy - H_i * h)

    return S

# End of file