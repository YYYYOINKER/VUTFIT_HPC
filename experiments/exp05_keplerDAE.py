import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sympy as sp
import numba as nb
from scipy.integrate import solve_ivp
from pyhamsys import HamSys, solve_ivp_sympext, Parameters

from numba import njit, float64, int64

# ── ODE system — DAE formulation (paper eq. 37) ───────────────────────────────
#
# Original Kepler problem:
#   y1' = y3,  y2' = y4,  y3' = -y1/r³,  y4' = -y2/r³
#
# Auxiliary ODE variables (y5..y8) remove division:
#   y5 = r³,  y6 = r,  y7 = 1/r³,  y8 = 1/r
#
# Algebraic variables (y9..y24) reduce all multiplications to bilinear:
#
#   Level 1 — depend only on y1..y8:
#     y9  = y1·y7     y10 = y2·y7
#     y11 = y1·y3     y12 = y2·y4
#     y13 = y7·y7     y14 = y8·y8
#
#   Level 2 — depend on level 1:
#     y15 = y6·y11    y16 = y6·y12
#     y17 = y8·y11    y18 = y8·y12
#     y19 = y6·y13    y20 = y8·y14
#
#   Level 3 — depend on level 2:
#     y21 = y11·y19   y22 = y12·y19
#     y23 = y11·y20   y24 = y12·y20
#
# Full DAE system (ODE part):
#   y1' =  y3
#   y2' =  y4
#   y3' = -y9
#   y4' = -y10
#   y5' =  3·y15 + 3·y16
#   y6' =  y17 + y18
#   y7' = -3·y21 - 3·y22
#   y8' = -y23 - y24
#
# State vector column mapping (TC has 24 columns):
#   0:y1  1:y2  2:y3  3:y4  4:y5  5:y6  6:y7  7:y8
#   8:y9  9:y10 10:y11 11:y12 12:y13 13:y14
#   14:y15 15:y16 16:y17 17:y18 18:y19 19:y20
#   20:y21 21:y22 22:y23 23:y24
#
# Key property: algebraic TC[n, 8..23] are computed via prod2 in dependency
# order (level1 -> level2 -> level3), then used to build ODE TC[k, 0..7].
# No prod3, no S_cache — everything lives in TC.
# ─────────────────────────────────────────────────────────────────────────────

# ── parameters ────────────────────────────────────────────────────────────────
order     = 50
t_span    = (0.0, np.pi * 5)
h         = math.pi / 100
tol_order = 1e-16
min_order = 2
e         = 0.75

N_STATE = 24   # total state size (8 ODE + 16 algebraic)
N_PHYS  = 8    # physical ODE variables used for convergence check & diagnostics

t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / h) + 1)

# ── precomputed lookup tables ─────────────────────────────────────────────────
BINOM = np.array([[math.comb(n, k) if k <= n else 0
                   for k in range(order + 1)]
                  for n in range(order + 1)], dtype=np.float64)
FACT  = np.array([math.factorial(k) for k in range(order + 1)], dtype=np.float64)
HPOW  = np.array([h**k              for k in range(order + 1)], dtype=np.float64)

# ── initial conditions ────────────────────────────────────────────────────────
y1_0 = 1.0 - e
y2_0 = 0.0
y3_0 = 0.0
y4_0 = math.sqrt((1.0 + e) / (1.0 - e))
r0   = math.sqrt(y1_0**2 + y2_0**2)

y5_0 = r0**3
y6_0 = r0
y7_0 = 1.0 / r0**3
y8_0 = 1.0 / r0

# Level 1 algebraic initial values
y9_0  = y1_0 * y7_0    # y1·y7
y10_0 = y2_0 * y7_0    # y2·y7
y11_0 = y1_0 * y3_0    # y1·y3  (= 0 at periapsis)
y12_0 = y2_0 * y4_0    # y2·y4  (= 0 at periapsis)
y13_0 = y7_0 * y7_0    # y7²
y14_0 = y8_0 * y8_0    # y8²

# Level 2 algebraic initial values
y15_0 = y6_0 * y11_0   # y6·y11
y16_0 = y6_0 * y12_0   # y6·y12
y17_0 = y8_0 * y11_0   # y8·y11
y18_0 = y8_0 * y12_0   # y8·y12
y19_0 = y6_0 * y13_0   # y6·y13
y20_0 = y8_0 * y14_0   # y8·y14

# Level 3 algebraic initial values
y21_0 = y11_0 * y19_0  # y11·y19
y22_0 = y12_0 * y19_0  # y12·y19
y23_0 = y11_0 * y20_0  # y11·y20
y24_0 = y12_0 * y20_0  # y12·y20

state_vector = np.array([
    y1_0,  y2_0,  y3_0,  y4_0,
    y5_0,  y6_0,  y7_0,  y8_0,
    y9_0,  y10_0, y11_0, y12_0,
    y13_0, y14_0, y15_0, y16_0,
    y17_0, y18_0, y19_0, y20_0,
    y21_0, y22_0, y23_0, y24_0,
], dtype=float)

# ── history storage ───────────────────────────────────────────────────────────
t_steps    = int((t_span[1] - t_span[0]) / h)
t_arr      = np.zeros(t_steps + 1)
hist       = np.zeros((t_steps + 1, N_STATE))
used_order = np.zeros(t_steps, dtype=int)

t_arr[0] = t_span[0]
hist[0]  = state_vector

# ── Leibniz product helper ────────────────────────────────────────────────────
# prod2(TC, a, b, n): n-th derivative of ya(t)·yb(t)
#   (f·g)^(n) = Σ_{j=0}^{n} C(n,j) · f^(j) · g^(n-j)
# TC has N_STATE columns — same function works for both ODE and algebraic cols.
@njit(float64(float64[:,:], int64, int64, int64, float64[:,:]), cache=True, fastmath=False)
def prod2(TC, a, b, n, BINOM):
    row = BINOM[n]
    s = 0.0
    for j in range(n + 1):
        s += row[j] * TC[j, a] * TC[n-j, b]
    return s

# ── DAE Taylor step ───────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False)
def taylor_step_dae(state_vector, TC, BINOM, FACT, HPOW, order, min_order, tol_order):
    """
    Single Taylor step for the 24-variable DAE Kepler system.

    Per k-iteration (n = k-1):
      1. Compute algebraic TC[n, 8..23] in dependency order (tree):
           Level 1: prod2 on ODE cols 0..7          → cols 8..13
           Level 2: prod2 on ODE + level-1 cols      → cols 14..19
           Level 3: prod2 on level-1 + level-2 cols  → cols 20..23
      2. Build ODE TC[k, 0..7] using algebraic TC[n, 8..23]
      3. Accumulate Taylor term and check convergence
    """
    TC[0] = state_vector
    new_state = state_vector.copy()
    used_ord  = order

    for k in range(1, order + 1):
        n = k - 1   # Leibniz convolution level

        # ── Level 1 algebraics at level n ────────────────────────────────────
        # Depend only on ODE cols 0..7, which are fully built for rows 0..n.
        # (TC[n, 0..7] was set when k=n in the previous iteration.)
        TC[n,  8] = prod2(TC, 0,  6, n, BINOM)   # y9  = y1 · y7
        TC[n,  9] = prod2(TC, 1,  6, n, BINOM)   # y10 = y2 · y7
        TC[n, 10] = prod2(TC, 0,  2, n, BINOM)   # y11 = y1 · y3
        TC[n, 11] = prod2(TC, 1,  3, n, BINOM)   # y12 = y2 · y4
        TC[n, 12] = prod2(TC, 6,  6, n, BINOM)   # y13 = y7 · y7
        TC[n, 13] = prod2(TC, 7,  7, n, BINOM)   # y14 = y8 · y8

        # ── Level 2 algebraics at level n ────────────────────────────────────
        # Depend on ODE cols + level-1 algebraic cols (8..13 just filled above).
        TC[n, 14] = prod2(TC,  5, 10, n, BINOM)  # y15 = y6 · y11
        TC[n, 15] = prod2(TC,  5, 11, n, BINOM)  # y16 = y6 · y12
        TC[n, 16] = prod2(TC,  7, 10, n, BINOM)  # y17 = y8 · y11
        TC[n, 17] = prod2(TC,  7, 11, n, BINOM)  # y18 = y8 · y12
        TC[n, 18] = prod2(TC,  5, 12, n, BINOM)  # y19 = y6 · y13
        TC[n, 19] = prod2(TC,  7, 13, n, BINOM)  # y20 = y8 · y14

        # ── Level 3 algebraics at level n ────────────────────────────────────
        # Depend on level-1 (cols 10,11) and level-2 (cols 18,19) just filled.
        TC[n, 20] = prod2(TC, 10, 18, n, BINOM)  # y21 = y11 · y19
        TC[n, 21] = prod2(TC, 11, 18, n, BINOM)  # y22 = y12 · y19
        TC[n, 22] = prod2(TC, 10, 19, n, BINOM)  # y23 = y11 · y20
        TC[n, 23] = prod2(TC, 11, 19, n, BINOM)  # y24 = y12 · y20

        # ── ODE recurrence: TC[k, 0..7] ──────────────────────────────────────
        # Uses algebraic TC[n, 8..23] just computed above.
        TC[k, 0] =  TC[k-1, 2]                          # y1' = y3
        TC[k, 1] =  TC[k-1, 3]                          # y2' = y4
        TC[k, 2] = -TC[n,  8]                           # y3' = -y9
        TC[k, 3] = -TC[n,  9]                           # y4' = -y10
        TC[k, 4] =  3.0*TC[n, 14] + 3.0*TC[n, 15]      # y5' =  3y15 + 3y16
        TC[k, 5] =      TC[n, 16] +     TC[n, 17]       # y6' =   y17 + y18
        TC[k, 6] = -3.0*TC[n, 20] - 3.0*TC[n, 21]      # y7' = -3y21 - 3y22
        TC[k, 7] =     -TC[n, 22] -     TC[n, 23]       # y8' =  -y23 - y24
        # TC[k, 8..23] will be filled next iteration when this k becomes n

        # ── accumulate Taylor term ────────────────────────────────────────────
        term = TC[k] * HPOW[k] / FACT[k]
        new_state += term

        # ── adaptive stopping: check convergence on physical vars 0..7 only ──
        # Algebraic vars converge together with physical ones.
        term_size = 0.0
        for v in range(N_PHYS):
            av = abs(term[v])
            if av > term_size:
                term_size = av

        if k >= min_order and term_size < tol_order:
            used_ord = k
            break

    return new_state, used_ord

# ── warmup ────────────────────────────────────────────────────────────────────
TC_dae = np.empty((order + 1, N_STATE), dtype=float)

print("JIT warmup...", end=" ", flush=True)
_sv = state_vector.copy()
_ = taylor_step_dae(_sv, TC_dae, BINOM, FACT, HPOW, order, min_order, tol_order)
print("done")

# ── main Taylor DAE loop ──────────────────────────────────────────────────────
t0_taylor = time.perf_counter()

for i in range(t_steps):
    state_vector, used_ord = taylor_step_dae(
        state_vector, TC_dae, BINOM, FACT, HPOW, order, min_order, tol_order)
    used_order[i] = used_ord
    t_arr[i + 1]  = t_arr[i] + h
    hist[i + 1]   = state_vector

t1_taylor   = time.perf_counter()
time_taylor = t1_taylor - t0_taylor

# ── scipy reference solvers (physical 4-var system) ──────────────────────────
def kepler_rhs(t, y):
    x, y_pos, vx, vy = y
    r3 = (x*x + y_pos*y_pos)**1.5
    return [vx, vy, -x/r3, -y_pos/r3]

y0_phys = [y1_0, y2_0, y3_0, y4_0]

t0 = time.perf_counter()
sol_rk45 = solve_ivp(kepler_rhs, t_span, y0_phys,
                     method='RK45', t_eval=t_eval, rtol=1e-13, atol=1e-14)
time_rk45 = time.perf_counter() - t0

t0 = time.perf_counter()
sol_dop = solve_ivp(kepler_rhs, t_span, y0_phys,
                    method='DOP853', t_eval=t_eval, rtol=1e-13, atol=1e-14)
time_dop = time.perf_counter() - t0

# ── pyhamsys ──────────────────────────────────────────────────────────────────
hs = HamSys(ndof=2)
H_kepler = lambda q1, q2, p1, p2, t: (p1**2 + p2**2)/2 - 1/sp.sqrt(q1**2 + q2**2)
hs.compute_vector_field(H_kepler)
y0_ham = np.array([y1_0, y2_0, y3_0, y4_0])

t0 = time.perf_counter()
sol_verlet = solve_ivp_sympext(hs, t_span, y0_ham,
                               Parameters(step=h, solver='Verlet'), t_eval=t_eval)
time_verlet = time.perf_counter() - t0

t0 = time.perf_counter()
sol_bm4 = solve_ivp_sympext(hs, t_span, y0_ham,
                             Parameters(step=h, solver='BM4'), t_eval=t_eval)
time_bm4 = time.perf_counter() - t0

# ── diagnostics ───────────────────────────────────────────────────────────────
def energy(x, y, vx, vy):
    return 0.5*(vx**2 + vy**2) - 1.0/np.sqrt(x**2 + y**2)

def ellipse_residual(x, y, e):
    if abs(e) < 1e-15:
        return np.abs(x**2 + y**2 - 1.0)
    return np.abs((x + e)**2 + y**2/(1.0 - e**2) - 1.0)

# extract physical vars from DAE hist (first 4 cols = x, y, vx, vy)
x_dae  = hist[:, 0]
y_dae  = hist[:, 1]
vx_dae = hist[:, 2]
vy_dae = hist[:, 3]

E_taylor = energy(x_dae,         y_dae,         vx_dae,        vy_dae)
E_rk45   = energy(sol_rk45.y[0], sol_rk45.y[1], sol_rk45.y[2], sol_rk45.y[3])
E_dop    = energy(sol_dop.y[0],  sol_dop.y[1],  sol_dop.y[2],  sol_dop.y[3])
E_verlet = energy(sol_verlet.y[0], sol_verlet.y[1], sol_verlet.y[2], sol_verlet.y[3])
E_bm4    = energy(sol_bm4.y[0],   sol_bm4.y[1],   sol_bm4.y[2],   sol_bm4.y[3])

dE_taylor = np.abs(E_taylor - E_taylor[0])
dE_rk45   = np.abs(E_rk45  - E_rk45[0])
dE_dop    = np.abs(E_dop   - E_dop[0])
dE_verlet = np.abs(E_verlet - E_verlet[0])
dE_bm4    = np.abs(E_bm4   - E_bm4[0])

R_taylor = ellipse_residual(x_dae,         y_dae,         e)
R_rk45   = ellipse_residual(sol_rk45.y[0], sol_rk45.y[1], e)
R_dop    = ellipse_residual(sol_dop.y[0],  sol_dop.y[1],  e)
R_verlet = ellipse_residual(sol_verlet.y[0], sol_verlet.y[1], e)
R_bm4    = ellipse_residual(sol_bm4.y[0],   sol_bm4.y[1],   e)

# ── auxiliary variable sanity check ──────────────────────────────────────────
# r³ stored in col 4 should match (x²+y²)^(3/2) recomputed from col 0,1
r3_stored    = hist[:, 4]
r3_recompute = (hist[:, 0]**2 + hist[:, 1]**2)**1.5
aux_err      = np.max(np.abs(r3_stored - r3_recompute))

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n── DAE Taylor settings ─────────────────────────────")
print(f"  system         = 24 variables (8 ODE + 16 algebraic)")
print(f"  max order      = {order}")
print(f"  min order      = {min_order}")
print(f"  order tol      = {tol_order:.1e}")
print(f"  eccentricity   = {e}")
print(f"  avg used order = {used_order.mean():.2f}")
print(f"  min/max order  = {used_order.min()} / {used_order.max()}")
print(f"  aux var error  = {aux_err:.2e}  (r³ consistency check)")

print(f"\n── Timing ───────────────────────────────")
print(f"  Taylor DAE (ord {order}):      {time_taylor:.3f} s")
print(f"  scipy RK45:                {time_rk45:.3f} s")
print(f"  scipy DOP853:              {time_dop:.3f} s")
print(f"  pyhamsys Verlet:           {time_verlet:.3f} s")
print(f"  pyhamsys BM4:              {time_bm4:.3f} s")

print(f"\n── Max |ΔE| ─────────────────────────────")
print(f"  Taylor DAE: {dE_taylor.max():.2e}")
print(f"  RK45:       {dE_rk45.max():.2e}")
print(f"  DOP853:     {dE_dop.max():.2e}")
print(f"  Verlet:     {dE_verlet.max():.2e}")
print(f"  BM4:        {dE_bm4.max():.2e}")

print(f"\n── Max ellipse residual ─────────────────")
print(f"  Taylor DAE: {R_taylor.max():.2e}")
print(f"  RK45:       {R_rk45.max():.2e}")
print(f"  DOP853:     {R_dop.max():.2e}")
print(f"  Verlet:     {R_verlet.max():.2e}")
print(f"  BM4:        {R_bm4.max():.2e}")

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(x_dae,           y_dae,           lw=1.5, label=f'Taylor DAE ≤{order}')
ax.plot(sol_rk45.y[0],   sol_rk45.y[1],   lw=1, ls='--',           label='RK45')
ax.plot(sol_dop.y[0],    sol_dop.y[1],    lw=1, ls=':',            label='DOP853')
ax.plot(sol_verlet.y[0], sol_verlet.y[1], lw=1, ls='-.',           label='Verlet')
ax.plot(sol_bm4.y[0],    sol_bm4.y[1],    lw=1, ls=(0,(3,1,1,1)), label='BM4')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Kepler orbit — DAE system (e={e})')
ax.set_aspect('equal'); ax.legend(); ax.grid(True)

ax = axes[1]
ax.semilogy(t_eval, dE_taylor+1e-20, lw=1.5, label=f'Taylor DAE ≤{order}')
ax.semilogy(t_eval, dE_rk45  +1e-20, lw=1, ls='--',           label='RK45')
ax.semilogy(t_eval, dE_dop   +1e-20, lw=1, ls=':',            label='DOP853')
ax.semilogy(t_eval, dE_verlet+1e-20, lw=1, ls='-.',           label='Verlet')
ax.semilogy(t_eval, dE_bm4   +1e-20, lw=1, ls=(0,(3,1,1,1)), label='BM4')
ax.set_xlabel('t'); ax.set_ylabel('|ΔE|')
ax.set_title('Energy drift'); ax.legend(); ax.grid(True)

ax = axes[2]
ax.semilogy(t_eval, R_taylor+1e-20, lw=1.5, label=f'Taylor DAE ≤{order}')
ax.semilogy(t_eval, R_rk45  +1e-20, lw=1, ls='--',           label='RK45')
ax.semilogy(t_eval, R_dop   +1e-20, lw=1, ls=':',            label='DOP853')
ax.semilogy(t_eval, R_verlet+1e-20, lw=1, ls='-.',           label='Verlet')
ax.semilogy(t_eval, R_bm4   +1e-20, lw=1, ls=(0,(3,1,1,1)), label='BM4')
ax.set_xlabel('t'); ax.set_ylabel('ellipse residual')
ax.set_title('Geometric orbit error'); ax.legend(); ax.grid(True)

plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t_arr[:-1], used_order, '.-')
plt.xlabel('t'); plt.ylabel('used Taylor order')
plt.title(f'Adaptive order — DAE (tol={tol_order:.0e}, e={e})')
plt.grid(True); plt.tight_layout(); plt.show()

labels = [f'Taylor\nDAE≤{order}', 'RK45', 'DOP853', 'Verlet\n(pyham)', 'BM4\n(pyham)']
times  = [time_taylor, time_rk45, time_dop, time_verlet, time_bm4]
plt.figure(figsize=(8, 4))
bars = plt.bar(labels, times, color=['steelblue','tomato','tomato','seagreen','seagreen'])
plt.bar_label(bars, fmt='%.3fs', padding=3)
plt.ylabel('Wall time (s)')
plt.title(f'Integration time  (t=[0,{t_span[1]:.2f}], h={h:.4f}, e={e})')
plt.tight_layout(); plt.show()

# ── dynamic ranking table ─────────────────────────────────────────────────────
methods = [
    {"name": f"Taylor DAE≤{order}", "time": float(time_taylor),
     "dE": float(dE_taylor.max()), "res": float(R_taylor.max())},
    {"name": "RK45",   "time": float(time_rk45),
     "dE": float(dE_rk45.max()),   "res": float(R_rk45.max())},
    {"name": "DOP853", "time": float(time_dop),
     "dE": float(dE_dop.max()),    "res": float(R_dop.max())},
    {"name": "Verlet", "time": float(time_verlet),
     "dE": float(dE_verlet.max()), "res": float(R_verlet.max())},
    {"name": "BM4",    "time": float(time_bm4),
     "dE": float(dE_bm4.max()),    "res": float(R_bm4.max())},
]

for key, field in [("rank_time","time"),("rank_dE","dE"),("rank_res","res")]:
    for rank, idx in enumerate(sorted(range(len(methods)),
                                      key=lambda i: methods[i][field]), start=1):
        methods[idx][key] = rank

taylor_time = methods[0]["time"]
taylor_dE   = methods[0]["dE"]
taylor_res  = methods[0]["res"]

for m in methods:
    m["speedup_vs_taylor"] = taylor_time / m["time"]
    m["dE_vs_taylor"]      = m["dE"] / taylor_dE
    m["res_vs_taylor"]     = m["res"] / taylor_res
    m["score"]             = m["rank_time"] + m["rank_dE"] + m["rank_res"]

overall_order = sorted(range(len(methods)),
    key=lambda i: (methods[i]["score"], methods[i]["rank_res"],
                   methods[i]["rank_dE"], methods[i]["rank_time"]))
for rank, idx in enumerate(overall_order, start=1):
    methods[idx]["rank_overall"] = rank

best_time = min(m["time"] for m in methods)
best_dE   = min(m["dE"]   for m in methods)
best_res  = min(m["res"]  for m in methods)

print("\n── Dynamic ranking table ──────────────────────────────────────────────────────────────")
header = (f"{'Method':<18}{'time [s]':>12}{'max |ΔE|':>14}{'max ellipse':>16}"
          f"{'spd/Tay':>10}{'dE/Tay':>10}{'res/Tay':>10}"
          f"{'r_t':>6}{'r_E':>6}{'r_R':>6}{'score':>8}{'ovr':>6}")
print(header)
print("-" * len(header))

for m in sorted(methods, key=lambda x: x["rank_overall"]):
    print(f"{m['name']:<18}"
          f"{m['time']:>12.4f}{'*' if m['time']==best_time else '':<1}"
          f"{m['dE']:>14.2e}{'*' if m['dE']==best_dE else '':<1}"
          f"{m['res']:>16.2e}{'*' if m['res']==best_res else '':<1}"
          f"{m['speedup_vs_taylor']:>10.2f}"
          f"{m['dE_vs_taylor']:>10.2f}"
          f"{m['res_vs_taylor']:>10.2f}"
          f"{m['rank_time']:>6}{m['rank_dE']:>6}{m['rank_res']:>6}"
          f"{m['score']:>8}{m['rank_overall']:>6}")

print("\n* = best in column")

print("\n── Ranking by time ─────────────────────")
for m in sorted(methods, key=lambda x: x["rank_time"]):
    print(f"{m['rank_time']}. {m['name']:<18}  {m['time']:.4f} s")

print("\n── Ranking by energy drift ─────────────")
for m in sorted(methods, key=lambda x: x["rank_dE"]):
    print(f"{m['rank_dE']}. {m['name']:<18}  {m['dE']:.2e}")

print("\n── Ranking by ellipse residual ─────────")
for m in sorted(methods, key=lambda x: x["rank_res"]):
    print(f"{m['rank_res']}. {m['name']:<18}  {m['res']:.2e}")
