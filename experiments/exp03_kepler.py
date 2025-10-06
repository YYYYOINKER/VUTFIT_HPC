#
#   Experiment 03: Kepler problem
#   Author: Pavol Mihalik, VUT FIT
#
#   Comparison of explicit Euler and symplectic Euler integration
#   in Hamiltonian form of the two-body problem.
#

import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib.animation as animation;

from src.systems.kepler import (
    explicit_euler,
    symplectic_euler,
    action,
    cumulative_action
)



#####################################
# Simulation parameters
#####################################

mu = 1.0;
z0 = np.array([0.7, 0.4, -0.3, 1.0]); # circular orbit start
t_span = (0.0, 25.0);
h = 0.02;

#####################################
# Integrate
#####################################

t_exp, z_exp = explicit_euler(z0, t_span, h, mu);
t_symp, z_symp = symplectic_euler(z0, t_span, h, mu);

S_exp = action(t_exp, z_exp, mu);
S_symp = action(t_symp, z_symp, mu);

print("Action (Explicit Euler):", S_exp);
print("Action (Symplectic Euler):", S_symp);

#####################################
# Plotting
#####################################

# --- 1. Cumulative Action ---
S_exp_series = cumulative_action(t_exp, z_exp, mu);
S_symp_series = cumulative_action(t_symp, z_symp, mu);

plt.figure(figsize=(6,4))
plt.plot(t_exp, S_exp_series, label="Explicit Euler", linestyle="--");
plt.plot(t_symp, S_symp_series, label="Symplectic Euler", linestyle="-");
plt.title("Cumulative Action over Time");
plt.xlabel("t");
plt.ylabel("S(t)");
plt.legend();
plt.tight_layout();
plt.show();


# --- 2. Orbit comparison ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5));

# Graph 1
axes[0].plot(z_exp[:, 0], z_exp[:, 1], "b--");
axes[0].set_title("Explicit euler (non symplectic)");
axes[0].set_xlabel("x");
axes[0].set_ylabel("y");
axes[0].set_aspect("equal", adjustable="box");
axes[0].grid(True);

# Graph 2
axes[1].plot(z_symp[:, 0], z_symp[:, 1], "orange");
axes[1].set_title("Symplectic euler.");
axes[1].set_xlabel("x");
axes[1].set_ylabel("y");
axes[1].set_aspect("equal", adjustable="box");
axes[1].grid(True);


plt.suptitle("Kepler problem - Phase-Spacae orbits", fontsize=14);
plt.tight_layout();
plt.show();

#####################################
# --- 3. Animation: Side-by-side orbit evolution ---
#####################################

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Common setup for both plots
for ax, title in zip(
    axes,
    ["Explicit Euler (non-symplectic)", "Symplectic Euler (structure-preserving)"]
):
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

# Initialize orbit trails (lines) and moving bodies (points)
line_exp, = axes[0].plot([], [], "b--", lw=1.5)
point_exp, = axes[0].plot([], [], "bo", markersize=6)

line_symp, = axes[1].plot([], [], color="orange", lw=1.5)
point_symp, = axes[1].plot([], [], "ro", markersize=6)

# Time text annotation
time_text = fig.text(0.45, 0.05, "", fontsize=12)

# Initialization function
def init():
    line_exp.set_data([], [])
    point_exp.set_data([], [])
    line_symp.set_data([], [])
    point_symp.set_data([], [])
    time_text.set_text("")
    return line_exp, point_exp, line_symp, point_symp, time_text

# Frame update function
def update(frame):
    # Explicit Euler
    line_exp.set_data(z_exp[:frame, 0], z_exp[:frame, 1])
    point_exp.set_data([z_exp[frame, 0]], [z_exp[frame, 1]])

    # Symplectic Euler
    line_symp.set_data(z_symp[:frame, 0], z_symp[:frame, 1])
    point_symp.set_data([z_symp[frame, 0]], [z_symp[frame, 1]])

    # Update time display
    time_text.set_text(f"t = {t_exp[frame]:.2f}")
    return line_exp, point_exp, line_symp, point_symp, time_text

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t_exp),
    init_func=init,
    interval=30,   # milliseconds per frame
    blit=False,    # disable blitting for smoothness
    repeat=True
)

plt.suptitle("Kepler Problem – Side-by-side Animation", fontsize=14)
plt.tight_layout()
plt.show()



