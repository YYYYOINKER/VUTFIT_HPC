import numpy as np;
import matplotlib.pyplot as plt;
import jax;
import jax.numpy as jnp;
from jax import grad, jit, vmap;
from jax import random;
import math;

#
# Purpose of this file is to try out different approaches to genearting
# higher derivatives for the Taylor serie method
# Problem: Potential well
# H = T + V
# T = p^2/2
# V = q^4/4 - q^2/2
# From the equations of motion
# q. = p/m, m=1 for simplicity
# p. = -q^3 + q
#

# Initial conditions
q0: float = -0.9;
p0: float = 0;
h: float = 1;
T: int = 10000*h;

# Functions from equations of motion
def H(y):
    q,p = y;
    return jnp.array([p, -q**3 + q]);

# Classic taylor expansion
def taylor_ad( q0:float, p0: float, h: float, T:int, order: int=10 ):

    # Setup arrays
    time_steps = int(T/h);
    t = np.zeros(time_steps);
    # y = [ [q0, p0], [q1, p1], ... [qn, pn] ]
    Y = np.zeros( (time_steps, 2) );
    # Initial conditions
    Y[0] = [q0, p0];

    # Small y for a specific point
    y = jnp.array([q0, p0]);

    # For all time steps
    for n in range(time_steps-1):

        # First derivative
        deriv = H(y);
        y_next = y + h * deriv;
        
        # Generate higher derivatives via autodiff
        for ord in range(2, order+1):
            
            # JVP - jacobian-vector-product,
            #parameters: (fun, primals, tangents)
            # fun function were derivating
            # point of evaluation
            # direction in which you derivate
            _, deriv = jax.jvp(H, (y,), (deriv,));
            y_next += (h**ord / math.factorial(ord)) * deriv;
        
        y = y_next;
        Y[n+1] = np.array(y);
    
    return Y;


# Running a test
Y = taylor_ad(q0, p0, h, T, 60);

plt.plot(Y[:,0], Y[:,1]);
plt.xlabel("q");
plt.ylabel("p");
plt.title("Double-well phase space Taylor");
plt.grid(True);
plt.show();

# End of file