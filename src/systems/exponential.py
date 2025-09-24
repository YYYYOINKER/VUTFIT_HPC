#
#   System for experiment 1
#   Author: Pavol Mihalik, VUT FIT
#
#   ODE: y' = λ y ; y(0) = 1
#   λ = lam in code
#

import numpy as np

# Function defining the ODE
def f(t: float, y: float, lam: float = 1.0) -> float:

    # Right hand side in y' = λ y
    return lam * y;

# Exact 'analytic' solution using numpy
def exact_solution(t: float, y0: float = 1.0, lam: float = 1.0) -> float:

    # Exact solution: y(t) = y0 * exp(λ t)
    return y0 * np.exp(lam * t);


##################### Specialized Taylorr Solver for this ODE #####################
#
# Higher taylor derivatives can be calculated as Di * h * λ / k ; where k is the order
#
# Parameters:
#   t_span : tuple (t0, t_max) 
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