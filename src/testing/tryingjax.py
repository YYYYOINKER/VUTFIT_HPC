import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# define a ugly function
f_scalar = lambda x: ( jnp.cos(x**2) * jnp.sin(x**2) / jnp.exp(2*x) );

# scalar -> scalar
f1_scalar = jax.grad(f_scalar);
f2_scalar = jax.grad(f1_scalar);
f3_scalar = jax.grad(f2_scalar);
f4_scalar = jax.grad(f3_scalar);
f5_scalar = jax.grad(f4_scalar);


# vectorize
f_vec = jax.jit(jax.vmap(f_scalar));
f_vec1 = jax.jit(jax.vmap(f1_scalar));
f_vec2 = jax.jit(jax.vmap(f2_scalar));
f_vec3 = jax.jit(jax.vmap(f3_scalar));
f_vec4 = jax.jit(jax.vmap(f4_scalar));
f_vec5 = jax.jit(jax.vmap(f5_scalar));

# create some x values to compure for
x = jnp.linspace(-3, 3, 1000);

# evaluate
y = f_vec(x);
y_1 = f_vec1(x);
y_2 = f_vec2(x);
y_3 = f_vec3(x);
y_4 = f_vec4(x);
y_5 = f_vec5(x);

x_np  = np.asarray(x);
plt.plot(x_np, np.asarray(y),   label="f");
plt.plot(x_np, np.asarray(y_1), label="f'");
plt.plot(x_np, np.asarray(y_2), label="f''");
plt.plot(x_np, np.asarray(y_3), label="f'''");
#plt.plot(x_np, np.asarray(y_4), label="f''''");
#plt.plot(x_np, np.asarray(y_5), label="f'''''");


plt.legend();
plt.show();

# End of file