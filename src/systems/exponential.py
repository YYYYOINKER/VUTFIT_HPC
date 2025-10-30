#
#   System for experiment 1
#   Author: Pavol Mihalik, VUT FIT
#
#   ODE: y' = λ y ; y(0) = 1
#   λ = lam in code
#
# Import general solvers

import numpy as np

# Function defining the ODE
def f(t: float, y: float, lam: float = 1.0) -> float:

    # Right hand side in y' = λ y
    return lam * y;

# Exact 'analy_tayloric' solution using numpy
def exact_solution(t: float, y0: float = 1.0, lam: float = 1.0) -> float:

    # Exact solution: y(t) = y0 * exp(λ t)
    return y0 * np.exp(lam * t);


def local_error_all_methods(f: callable, t_span: tuple[float, float], y0: float = 1.0, lam: float = 1.0, h: float = 0.1
                      ) -> tuple[np.ndarray, np.ndarray]:

    # Setup
    t_steps:int = int( (t_span[1] - t_span[0]) / h );
    t_starts:np.ndarray = t_span[0] + np.arange(t_steps) * h; # ti, t1, t2, ...
    lte_euler:np.ndarray = np.zeros(t_steps);
    lte_rk2:np.ndarray = np.zeros(t_steps);
    lte_rk4:np.ndarray = np.zeros(t_steps);
    lte_taylor:np.ndarray = np.zeros(t_steps);

    order: int = 10;

    # i - index of element, ti - time at that index in t_points
    for i, ti in enumerate(t_starts):
        
        # exact state at the point
        y_exact_start = exact_solution(ti, y0=y0, lam=lam);

        # --- euler step ---
        y_num_euler = y_exact_start + h * f(ti, y_exact_start, lam);

        # --- rk2 step ---
        k1 = f(ti,     y_exact_start,          lam); # slope at start
        k2 = f(ti + h, y_exact_start + h * k1, lam); # slope at end
        y_num_rk2 = y_exact_start + h * 0.5 * (k1 + k2); # final calculatoin

        # --- rk4 step ---
        k1 = f(ti,           y_exact_start,                lam);
        k2 = f(ti + 0.5 * h, y_exact_start + 0.5 * h * k1, lam);
        k3 = f(ti + 0.5 * h, y_exact_start + 0.5 * h * k2, lam);
        k4 = f(ti + h,       y_exact_start + h * k3, lam);
        y_num_rk4 = y_exact_start + h/6 * (k1 + 2 * k2 + 2 * k3 + k4);

        # --- taylor ---
        term = y_exact_start;
        y_num_taylor = term;

        # Taylor expansion up to desired order (ord)
        for j in range(1, order+1):
            term = term * h * lam / j;
            y_num_taylor += term;

        # end point at ti+h
        y_exact_end = exact_solution(ti+h, y0=y0, lam=lam);

        # get local truncuation error wia: approx solution in point - exact
        lte_euler[i] = abs(y_num_euler - y_exact_end);
        lte_rk2[i] = abs(y_num_rk2 - y_exact_end);
        lte_rk4[i] = abs(y_num_rk4 - y_exact_end);
        lte_taylor[i] = abs(y_num_taylor - y_exact_end);

    # return a stack of these arrays
    ltes = np.stack([lte_euler, lte_rk2, lte_rk4, lte_taylor], axis=0)

    return t_starts, ltes;

def global_error_all_methods(f: callable, t_span: tuple[float, float], y0: float = 1.0, lam: float = 1.0, h: float = 0.1
                      ) -> tuple[np.ndarray, np.ndarray]:

    # Setup
    t0, t1 = t_span;
    t_steps = int((t1 - t0)/h);
    t = t0 + np.arange(t_steps+1)*h;

    # Error arrays
    gerr_euler:np.ndarray = np.zeros(t_steps+1);
    gerr_rk2:np.ndarray = np.zeros(t_steps+1);
    gerr_rk4:np.ndarray = np.zeros(t_steps+1);
    gerr_taylor:np.ndarray = np.zeros(t_steps+1);

    # Initial conditions for al lstates
    y_euler = y_rk2 = y_rk4 = y_taylor = y0;

    # exact
    y_exact = exact_solution(t, y0=y0, lam=lam);
    order: int = 10;

    for i in range(t_steps):

        # --- euler ---
        y_euler = y_euler + h * f(t[i], y_euler, lam);

        # --- rk2 ---
        k1 = f(t[i], y_rk2, lam);
        k2 = f(t[i] + h, y_rk2 + h*k1, lam);
        y_rk2 = y_rk2 + 0.5*h*(k1 + k2);

        # --- rk4 ---
        k1 = f(t[i], y_rk4, lam)
        k2 = f(t[i] + 0.5*h, y_rk4 + 0.5*h*k1, lam);
        k3 = f(t[i] + 0.5*h, y_rk4 + 0.5*h*k2, lam);
        k4 = f(t[i] + h,     y_rk4 + h*k3,     lam);
        y_rk4 = y_rk4 + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4);

        # --- taylor ---
        s = 1.0; term = 1.0;
        for j in range(1, order+1):
            term *= (lam*h)/j;
            s += term;
        y_taylor = y_taylor * s;

        # global error arrays
        gerr_euler[i+1] = abs(y_euler - y_exact[i+1]);
        gerr_rk2[i+1] = abs(y_rk2 - y_exact[i+1]);
        gerr_rk4[i+1] = abs(y_rk4 - y_exact[i+1]);
        gerr_taylor[i+1] = abs(y_taylor - y_exact[i+1]);

    errs = np.stack([gerr_euler, gerr_rk2, gerr_rk4, gerr_taylor], axis=0);
    return t, errs;


##################### Specialized Taylorr Solver for this ODE #####################
#
# Higher taylor derivatives can be calculated as Di * h * λ / k ; where k is the order
#
# Parameters:
#   t_span : tuple (ti, t_max) 
#       - Time interval
#
#   y0 : float 
#       - Initial condition
#
#   h: float 
#       - Step size
#
#   lam : float 
#       - λ parameter
#
#   order : int
#       - Desired Taylor series expansion order
#
#   Returns:
#   t : np.ndarray
#       - Time grid
#
#   y : np.ndarray
#       - Approximated solution
#

def taylor_recursive_diff(t_span: tuple[float, float], 
                          y0: float, 
                          h: float, 
                          lam: float = 1.0, 
                          order: int = 10
                          ) -> tuple[np.ndarray, np.ndarray]:

    # Setup
    t_steps:int = int( (t_span[1] - t_span[0]) / h );
    t:np.ndarray = np.zeros(t_steps + 1);
    y:np.ndarray = np.zeros(t_steps + 1);
    y[0] = y0;
    t[0] = t_span[0];

    # Taylor steps
    for i in range(t_steps):

        term = y[i];
        y_next = term;

        # Taylor expansion up to desired order (ord)
        for j in range(1, order+1):
            term = term * h * lam / j;
            y_next += term;
        
        # Update arrays
        y[i+1] = y_next;
        t[i+1] = t[i]+h;
    
    return t, y;

# End of file