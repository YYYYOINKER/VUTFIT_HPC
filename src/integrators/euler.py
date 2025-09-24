#
#   General Euler solver
#   Author: Pavol Mihalik, VUT FIT
#

import numpy as np;

#
# Parameters:
#   f : 'function pointer'
#       - defines right hand side of ODE
#
#   t_span : tuple (t0, t_max) 
#       - Time interval
#
#   y0 : float 
#       - Initial condition
#
#   h: float 
#       - Step size
#
#   Returns:
#   t : np.ndarray
#       - Time grid
#
#   y : np.ndarray
#       - Approximated solution
#

def euler(f: callable, t_span: tuple[float, float], y0: float, h:float
          ) -> tuple[np.ndarray, np.ndarray]:

    # Setup
    t_steps:int = int( (t_span[1] - t_span[0]) / h );
    t:np.ndarray = np.zeros(t_steps + 1);
    y:np.ndarray = np.zeros(t_steps + 1);
    y[0] = y0;
    t[0] = t_span[0];

    # Euler loop
    for i in range(t_steps):
        y[i+1] = y[i] + h * f(t[i], y[i]);
        t[i+1] = t[i]+h;

    return t, y;

# End of file