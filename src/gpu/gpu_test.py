#
# WIP learning PyopenCL
#
import pyopencl as cl
import numpy as np

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
__kernel void double_array(__global float* a) {
    int grid = get_global_id(0);
    a[grid] *= 2.0f;
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
