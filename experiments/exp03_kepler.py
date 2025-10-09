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
    cumulative_action,
    H
)


#####################################
# Simulation parameters
#####################################

mu = 1.0;
z0 = np.array([1.0, 0.0, 0.0, -1.0]); # circular orbit start [0.7, 0.4, -0.3, 1.0]
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

fig, axes = plt.subplots(1, 2, figsize=(10, 5));

# Common setup for both plots
for ax, title in zip(
    axes,
    ["Explicit Euler (non-symplectic)", "Symplectic Euler (structure-preserving)"]
):
    ax.set_xlim(-5, 5);
    ax.set_ylim(-5, 5);
    ax.set_aspect("equal", adjustable="box");
    ax.set_title(title);
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.grid(True);

# Initialize orbit trails (lines) and moving bodies (points)
line_exp, = axes[0].plot([], [], "b--", lw=1.5);
point_exp, = axes[0].plot([], [], "bo", markersize=6);

line_symp, = axes[1].plot([], [], color="orange", lw=1.5);
point_symp, = axes[1].plot([], [], "ro", markersize=6);

# Time text annotation
time_text = fig.text(0.45, 0.05, "", fontsize=12);

# Initialization function
def init():
    line_exp.set_data([], []);
    line_symp.set_data([], []);
    point_symp.set_data([], []);
    time_text.set_text("");
    return line_exp, point_exp, line_symp, point_symp, time_text;

# Frame update function
def update(frame):
    # Explicit Euler
    line_exp.set_data(z_exp[:frame, 0], z_exp[:frame, 1]);
    point_exp.set_data([z_exp[frame, 0]], [z_exp[frame, 1]]);

    # Symplectic Euler
    line_symp.set_data(z_symp[:frame, 0], z_symp[:frame, 1]);
    point_symp.set_data([z_symp[frame, 0]], [z_symp[frame, 1]]);

    # Update time display
    time_text.set_text(f"t = {t_exp[frame]:.2f}");
    return line_exp, point_exp, line_symp, point_symp, time_text;

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t_exp),
    init_func=init,
    interval=5,   # milliseconds per frame
    blit=False,    # disable blitting for smoothness
    repeat=True
);

plt.suptitle("Kepler Problem – Side-by-side Animation", fontsize=14);
plt.tight_layout();
plt.show();

#####################################
# --- 4. Animation: actual coordinates of BOTH masses (COM frame) ---
#####################################

# masses with G=1 and m1+m2 == mu
m1, m2 = 0.6, 0.4
assert abs((m1 + m2) - mu) < 1e-12, "Choose masses so that m1+m2 == mu"

# Relative positions r = (x,y)
r_exp  = z_exp[:, :2]
r_symp = z_symp[:, :2]

# Convert to COM positions: r1 = -(m2/(m1+m2)) r, r2 = (m1/(m1+m2)) r
fac1 = -m2 / (m1 + m2)
fac2 =  m1 / (m1 + m2)
r1_exp,  r2_exp  = fac1 * r_exp,  fac2 * r_exp
r1_symp, r2_symp = fac1 * r_symp, fac2 * r_symp

# Frames (use the shortest)
N = min(len(t_exp), len(t_symp))

# Axis limits
all_xy = np.vstack([r1_exp[:N], r2_exp[:N], r1_symp[:N], r2_symp[:N]])
Rmax = np.nanmax(np.linalg.norm(all_xy, axis=1))
pad = 0.20*Rmax if Rmax > 0 else 1.0
xmin, ymin = np.nanmin(all_xy, axis=0) - pad
xmax, ymax = np.nanmax(all_xy, axis=0) + pad

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))
titles = ["Explicit Euler (actual masses)", "Symplectic Euler (actual masses)"]
for ax, title in zip(axes4, titles):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True); ax.set_title(title)
    ax.plot([0],[0], marker="x", markersize=6)  # COM marker

# Trailing lines
(line1_exp,)  = axes4[0].plot([], [], lw=1.2, linestyle="-",  label="mass 1")
(line2_exp,)  = axes4[0].plot([], [], lw=1.2, linestyle="--", label="mass 2")
(line1_symp,) = axes4[1].plot([], [], lw=1.2, linestyle="-",  label="mass 1")
(line2_symp,) = axes4[1].plot([], [], lw=1.2, linestyle="--", label="mass 2")
axes4[0].legend(loc="upper right")
axes4[1].legend(loc="upper right")

# Moving bodies as scatter (more robust than plot(..., "o"))
pt1_exp  = axes4[0].scatter([], [], s=25)
pt2_exp  = axes4[0].scatter([], [], s=25)
pt1_symp = axes4[1].scatter([], [], s=25)
pt2_symp = axes4[1].scatter([], [], s=25)

time_text4 = fig4.text(0.45, 0.05, "", fontsize=12)

def init4():
    # Populate t=0 so the figure is not empty even if animation doesn't start
    line1_exp.set_data(r1_exp[:1,0], r1_exp[:1,1])
    line2_exp.set_data(r2_exp[:1,0], r2_exp[:1,1])
    pt1_exp.set_offsets(r1_exp[0:1, :])
    pt2_exp.set_offsets(r2_exp[0:1, :])

    line1_symp.set_data(r1_symp[:1,0], r1_symp[:1,1])
    line2_symp.set_data(r2_symp[:1,0], r2_symp[:1,1])
    pt1_symp.set_offsets(r1_symp[0:1, :])
    pt2_symp.set_offsets(r2_symp[0:1, :])

    time_text4.set_text(f"t = {t_exp[0]:.2f}")
    return (line1_exp, line2_exp, pt1_exp, pt2_exp,
            line1_symp, line2_symp, pt1_symp, pt2_symp, time_text4)

def update4(i):
    # Explicit Euler
    line1_exp.set_data(r1_exp[:i+1,0], r1_exp[:i+1,1])
    line2_exp.set_data(r2_exp[:i+1,0], r2_exp[:i+1,1])
    pt1_exp.set_offsets(r1_exp[i:i+1, :])
    pt2_exp.set_offsets(r2_exp[i:i+1, :])

    # Symplectic Euler
    line1_symp.set_data(r1_symp[:i+1,0], r1_symp[:i+1,1])
    line2_symp.set_data(r2_symp[:i+1,0], r2_symp[:i+1,1])
    pt1_symp.set_offsets(r1_symp[i:i+1, :])
    pt2_symp.set_offsets(r2_symp[i:i+1, :])

    time_text4.set_text(f"t = {t_exp[i]:.2f}")
    return (line1_exp, line2_exp, pt1_exp, pt2_exp,
            line1_symp, line2_symp, pt1_symp, pt2_symp, time_text4)

# Draw first frame immediately (helps some backends)
init4()
fig4.canvas.draw_idle()

ani4 = animation.FuncAnimation(
    fig4, update4, frames=N, init_func=init4,
    interval=40, blit=False, repeat=True, repeat_delay=800
)

plt.suptitle("Two-body motion (COM frame) – Explicit vs Symplectic", fontsize=14)
plt.tight_layout()
plt.show()

# End of file