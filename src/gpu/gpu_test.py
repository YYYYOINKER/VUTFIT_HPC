#
# WIP learning PyopenCL
#
import pyopencl as cl;
import numpy as np;

# Create OpenCL Context
# Choose a device
ctx = cl.create_some_context(interactive=True);

# Command queue to submit commands
queue = cl.CommandQueue(ctx);
print("Queue created: ", queue);

# Data
a = np.arange(10, dtype=np.float32);
print("Input: ", a);

# Kernel code
kernel_code = """
    // testing C stuff...
    // no dynamic memory allowed
    // no pointers
    // structs are ok although data should be manippulated through python

    #define is_positive(x) ((x) >= 0 ? 1 : 0)

    typedef struct {
            float value;
            float derivative;
        } node_t;

__kernel void double_array(__global float* a) {

    // get id
    int grid = get_global_id(0);

    // double array
    a[grid] *= 2.0f;

    // if elemnt in grid array is positive
    if ( is_positive(a[grid]) ) {

         // set to 1
         a[grid] = 1.0f;
    }
}
"""

# Make program ??
program = cl.Program(ctx, kernel_code).build();

# Setup buffers
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a);

# Run kernel
program.double_array(queue, a.shape, None, a_buf);

cl.enqueue_copy(queue, a, a_buf);
print("Output: ", a);
