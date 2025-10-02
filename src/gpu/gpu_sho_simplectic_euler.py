#
# Mass on a spring
# H(p, q) = p^2 / 2m  +  kq^2 / 2
# q˙​ = ∂H/∂p  <=> p/m
# p˙ = -∂H/∂q <=> -kq

import numpy as np;
import matplotlib.pyplot as plt;
import pyopencl as cl;
import random;

# Context and queue
ctx = cl.create_some_context(interactive=False);
queue = cl.CommandQueue(ctx);

# Parameters
n: int = 1000; # Number of oscillators
dt: float = 0.01;
k: float = 1.0;
m: float = 1.0;
steps: int = 100;

# Initial conditions randomly selected 
q: np.ndarray = np.random.uniform(-1, 1, size=n).astype(np.float32); # np.ndarray of floats from -1 - 1
p: np.ndarray = np.zeros_like(q);

# Kernel
kernel = '''
__kernel void simplectic_euler_step(

    __global float* q,
    __global float* p,
    float dt, float k, float m, int steps ) {

        int i = get_global_id(0);
        float q = q_out[i]; 
        float p = p_out[i];

        for (int s = 0; s < steps; s++) {

            // simplectic Euler update
            p = p - dt * k * q;
            q = q + dt * p / m;

            // write snapshot
            q_out[s*n + i] = q;
            p_out[s*n + i] = p;
        }
    }
'''

# Build program
program = cl.Program(ctx, kernel).build();
mf = cl.mem_flags;

# Retrieve kernel once
simplectic_step = cl.Kernel(program, "simplectic_euler_step");

# Buffers
q_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=q)
p_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=p)

# Run integration
for _ in range(steps):
    simplectic_step.set_args(
        q_buf, p_buf,
        np.float32(dt), np.float32(k), np.float32(m), steps );
    cl.enqueue_nd_range_kernel(queue, simplectic_step, (n,), None);

# Return results back
cl.enqueue_copy(queue, q, q_buf);
cl.enqueue_copy(queue, p, p_buf);

print("Final positions:", q[:10]);
print("Final momenta:", p[:10]);

energy = 0.5 * (p**2 / m + k * q**2);
print("Sample energies:", energy[:10]);
print("Mean energy:", energy.mean());



# End of file