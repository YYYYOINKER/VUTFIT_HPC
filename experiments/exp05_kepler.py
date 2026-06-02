import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sympy as sp
import numba as nb
from scipy.integrate import solve_ivp
from pyhamsys import HamSys, solve_ivp_sympext, Parameters

#from numba import njit, jit, prange, float64, int64
from numba import njit as _njit, float64, int64
def njit(*args, **kwargs):
    def decorator(fn):
        return fn
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

# OPTIMIZATIONS
# - precompute binomial coefficients instead of using math.comb(...) directly in loop
# - precompute factorials instead of using math.factorial(k) directly in loop
# - precompute h**k
# - memoization of nonlinear taylor terms via S_cache, r_invr3_sq, invr_cubed
# TODO
# 
# rozisrna aritmetika double double 
# porovnat max roder 4 s RK4
# adaptivny krok
# DAE KEPLER

# ── parameters ────────────────────────────────────────────────────────────────
order     = 40
t_span    = (0.0, np.pi * 16)
h         = math.pi / 200
tol_order = 1e-12
min_order = 2
e         = 0.25   # eccentricity; use e=0.0 for circular orbit

t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / h) + 1)

# ── precomputed lookup tables ─────────────────────────────────────────────────
# Avoids calling math.comb / math.factorial / pow inside the hot loop.
BINOM = np.array([[math.comb(n, k) if k <= n else 0
                   for k in range(order + 1)]
                  for n in range(order + 1)], dtype=np.float64)
FACT  = np.array([math.factorial(k) for k in range(order + 1)], dtype=np.float64)
HPOW  = np.array([h**k              for k in range(order + 1)], dtype=np.float64)

# ── initial conditions (Kepler ellipse at periapsis) ──────────────────────────
# State vector layout (8 components):
#   0: x      – position x
#   1: y      – position y
#   2: vx     – velocity x
#   3: vy     – velocity y
#   4: r3     – r^3 = (x²+y²)^(3/2)   removes r^3 from denominator of force
#   5: r      – r   = (x²+y²)^(1/2)   auxiliary for r3 derivative
#   6: inv_r3 – 1/r³                   makes force term bilinear: vx' = -x * inv_r3
#   7: inv_r  – 1/r                    auxiliary for inv_r3 derivative
y1_0 = 1.0 - e
y2_0 = 0.0
y3_0 = 0.0
y4_0 = math.sqrt((1.0 + e) / (1.0 - e))

r0 = math.sqrt(y1_0**2 + y2_0**2)

state_vector = np.array([
    y1_0, y2_0, y3_0, y4_0,
    r0**3,      # r3
    r0,         # r
    1.0/r0**3,  # inv_r3
    1.0/r0,     # inv_r
], dtype=float)

# ── history storage ───────────────────────────────────────────────────────────
t_steps    = int((t_span[1] - t_span[0]) / h)
t_arr      = np.zeros(t_steps + 1)
hist       = np.zeros((t_steps + 1, 8))
used_order = np.zeros(t_steps, dtype=int)

t_arr[0] = t_span[0]
hist[0]  = state_vector

# ── A and B matrices ──────────────────────────────────────────────────────────
# ODE system:  y' = A @ y + B @ nonlin
#
# A handles the two linear equations:
#   x'  = vx   (col 0 ← col 2)
#   y'  = vy   (col 1 ← col 3)
#
# nonlin vector layout (6 entries):
#   [0] x   * inv_r3          → vx'     = -x/r³
#   [1] y   * inv_r3          → vy'     = -y/r³
#   [2] r   * S               → r3'     =  3·r·S
#   [3] inv_r * S             → r'      =  inv_r·S
#   [4] r * inv_r3² * S       → inv_r3' = -3·r·inv_r3²·S
#   [5] inv_r³ * S            → inv_r'  = -inv_r³·S
# where S = x·vx + y·vy
A = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],   # x'  = vx
    [0, 0, 0, 1, 0, 0, 0, 0],   # y'  = vy
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)

B = np.array([
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0],   # vx'     = -nonlin[0]
    [ 0, -1,  0,  0,  0,  0],   # vy'     = -nonlin[1]
    [ 0,  0,  3,  0,  0,  0],   # r3'     =  3*nonlin[2]
    [ 0,  0,  0,  1,  0,  0],   # r'      =    nonlin[3]
    [ 0,  0,  0,  0, -3,  0],   # inv_r3' = -3*nonlin[4]
    [ 0,  0,  0,  0,  0, -1],   # inv_r'  =   -nonlin[5]
], dtype=float)

# ── Leibniz product helpers ───────────────────────────────────────────────────
# TC[k, col] = k-th Taylor coefficient of state variable col.
#
# prod2(TC, a, b, n): n-th derivative of  ya(t) * yb(t)
#   (f·g)^(n) = Σ_{j=0}^{n} C(n,j) · f^(j) · g^(n-j)
@njit(float64(float64[:,:], int64, int64, int64, float64[:,:]), cache=True, fastmath=False)
def prod2(TC, a, b, n, BINOM):
    row = BINOM[n]
    s = 0.0
    for j in range(n + 1):
        s += row[j] * TC[j, a] * TC[n-j, b]
    return s

# prod3(TC, a, b, c, n): n-th derivative of  ya(t) * yb(t) * yc(t)
#   double Leibniz: split as ya * (yb*yc), then split yb*yc again
@njit(float64(float64[:,:], int64, int64, int64, int64, float64[:,:]), cache=True, fastmath=False)
def prod3(TC, a, b, c, n, BINOM):
    row_n = BINOM[n]
    s = 0.0
    for j in range(n + 1):
        row_nj = BINOM[n - j]
        tc_ja  = TC[j, a]
        for m in range(n - j + 1):
            s += row_n[j] * row_nj[m] * tc_ja * TC[m, b] * TC[n-j-m, c]
    return s

# S_deriv(TC, n): n-th derivative of S(t) = x(t)·vx(t) + y(t)·vy(t)
@njit(float64(float64[:,:], int64, float64[:,:]), cache=True, fastmath=False)
def S_deriv(TC, n, BINOM):
    return prod2(TC, 0, 2, n, BINOM) + prod2(TC, 1, 3, n, BINOM)

@njit(cache=True, fastmath=False)
def taylor_step_jit(state_vector, TC, S_cache, r_invr3_sq, invr_cubed,
                    x_invr3, y_invr3, BINOM, FACT, HPOW, A, B,
                    order, min_order, tol_order):

    TC[0] = state_vector

    # seed level-0 cache
    S_cache[0]    = S_deriv(TC, 0, BINOM)
    r_invr3_sq[0] = prod3(TC, 5, 6, 6, 0, BINOM)
    invr_cubed[0] = prod3(TC, 7, 7, 7, 0, BINOM)
    x_invr3[0]    = prod2(TC, 0, 6, 0, BINOM)
    y_invr3[0]    = prod2(TC, 1, 6, 0, BINOM)

    new_state = state_vector.copy()
    used_ord  = order

    for k in range(1, order + 1):
        n = k - 1

        # leibnitz sum caching for later use
        #   x'      =  vx
        #   y'      =  vy
        #   vx'     = -x  · (1/r³)
        #   vy'     = -y  · (1/r³)
        #   (r³)'   =  3  · r     · S          S = x·vx + y·vy
        #   r'      =       (1/r) · S
        #   (1/r³)' = -3  · r     · (1/r³)² · S
        #   (1/r)'  =      -(1/r)³            · S
        S_cache[n]    = S_deriv(TC, n, BINOM)
        r_invr3_sq[n] = prod3(TC, 5, 6, 6, n, BINOM)
        invr_cubed[n] = prod3(TC, 7, 7, 7, n, BINOM)
        x_invr3[n]    = prod2(TC, 0, 6, n, BINOM)
        y_invr3[n]    = prod2(TC, 1, 6, n, BINOM)

        nonlin = np.zeros(6)
        nonlin[0] = x_invr3[n]
        nonlin[1] = y_invr3[n]

        row = BINOM[n]
        # are simmilarr
        for j in range(n + 1):
            # lookup
            c   = row[j]
            Snj = S_cache[n - j]

            # Leibniz sum
            nonlin[2] += c * TC[j, 5]       * Snj
            nonlin[3] += c * TC[j, 7]       * Snj
            nonlin[4] += c * r_invr3_sq[j]  * Snj
            nonlin[5] += c * invr_cubed[j]  * Snj

        # explicit new taylor term
        TC[k, 0] = TC[k-1, 2]
        TC[k, 1] = TC[k-1, 3]
        TC[k, 2] = -nonlin[0]
        TC[k, 3] = -nonlin[1]
        TC[k, 4] =  3.0 * nonlin[2]
        TC[k, 5] =        nonlin[3]
        TC[k, 6] = -3.0 * nonlin[4]
        TC[k, 7] =       -nonlin[5]

       # TC[k] = A @ TC[k - 1] + B @ nonlin

        # add computed taylor term to current step
        term      = TC[k] * HPOW[k] / FACT[k]
        new_state += term

        term_size = 0.0
        # term_size of only real 4 variables x y vx vy the auxiliary ones converge with them
        for v in range(8):
            av = abs(term[v])
            if av > term_size:
                term_size = av

        if k >= min_order and term_size < tol_order:
            used_ord = k
            break

    return new_state, used_ord

# ── Taylor integration loop ───────────────────────────────────────────────────
t0_taylor = time.perf_counter()

TC         = np.empty((order + 1, 8), dtype=float) # Taylor coefficients
S_cache    = np.empty(order, dtype=float)   # S_cache[m]    = m-th deriv of S = x·vx + y·vy
r_invr3_sq = np.empty(order, dtype=float)   # r_invr3_sq[m] = m-th deriv of r · (1/r³)²
invr_cubed = np.empty(order, dtype=float)   # invr_cubed[m] = m-th deriv of (1/r)³

# also cache the two prod2 terms that appear in nonlin[0] and nonlin[1]
x_invr3    = np.empty(order, dtype=float)   # x_invr3[m]    = m-th deriv of x · (1/r³)
y_invr3    = np.empty(order, dtype=float)   # y_invr3[m]    = m-th deriv of y · (1/r³)


# main loop
print("JIT warmup...", end=" ", flush=True)
_sv = state_vector.copy()
_ = taylor_step_jit(_sv, TC, S_cache, r_invr3_sq, invr_cubed,
                    x_invr3, y_invr3, BINOM, FACT, HPOW, A, B,
                    order, min_order, tol_order)
print("done")

# ── main Taylor loop — JIT ────────────────────────────────────────────────────
t0_taylor = time.perf_counter()

for i in range(t_steps):
    state_vector, used_ord = taylor_step_jit(
        state_vector, TC, S_cache, r_invr3_sq, invr_cubed,
                    x_invr3, y_invr3, BINOM, FACT, HPOW, A, B,
                    order, min_order, tol_order)
    used_order[i] = used_ord
    t_arr[i + 1]  = t_arr[i] + h
    hist[i + 1]   = state_vector

t1_taylor   = time.perf_counter()
time_taylor = t1_taylor - t0_taylor

# ── scipy RK45 / DOP853 ───────────────────────────────────────────────────────
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

# ── pyhamsys: Verlet (order 2) and BM4 (order 4) ─────────────────────────────
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

E_taylor = energy(hist[:,0], hist[:,1], hist[:,2], hist[:,3])
E_rk45   = energy(sol_rk45.y[0], sol_rk45.y[1], sol_rk45.y[2], sol_rk45.y[3])
E_dop    = energy(sol_dop.y[0],  sol_dop.y[1],  sol_dop.y[2],  sol_dop.y[3])
E_verlet = energy(sol_verlet.y[0], sol_verlet.y[1], sol_verlet.y[2], sol_verlet.y[3])
E_bm4    = energy(sol_bm4.y[0],   sol_bm4.y[1],   sol_bm4.y[2],   sol_bm4.y[3])

dE_taylor = np.abs(E_taylor - E_taylor[0])
dE_rk45   = np.abs(E_rk45  - E_rk45[0])
dE_dop    = np.abs(E_dop   - E_dop[0])
dE_verlet = np.abs(E_verlet - E_verlet[0])
dE_bm4    = np.abs(E_bm4   - E_bm4[0])

R_taylor = ellipse_residual(hist[:,0], hist[:,1], e)
R_rk45   = ellipse_residual(sol_rk45.y[0], sol_rk45.y[1], e)
R_dop    = ellipse_residual(sol_dop.y[0],  sol_dop.y[1],  e)
R_verlet = ellipse_residual(sol_verlet.y[0], sol_verlet.y[1], e)
R_bm4    = ellipse_residual(sol_bm4.y[0],   sol_bm4.y[1],   e)

# ── summary prints ────────────────────────────────────────────────────────────
print(f"\n── Taylor adaptive-order settings ─────────────────")
print(f"  max order      = {order}")
print(f"  min order      = {min_order}")
print(f"  order tol      = {tol_order:.1e}")
print(f"  eccentricity   = {e}")
print(f"  avg used order = {used_order.mean():.2f}")
print(f"  min used order = {used_order.min()}")
print(f"  max used order = {used_order.max()}")

print(f"\n── Timing ───────────────────────────────")
print(f"  Taylor adaptive (max ord {order}):  {time_taylor:.3f} s")
print(f"  scipy RK45:                         {time_rk45:.3f} s")
print(f"  scipy DOP853:                       {time_dop:.3f} s")
print(f"  pyhamsys Verlet:                    {time_verlet:.3f} s")
print(f"  pyhamsys BM4:                       {time_bm4:.3f} s")

print(f"\n── Max |ΔE| ─────────────────────────────")
print(f"  Taylor:   {dE_taylor.max():.2e}")
print(f"  RK45:     {dE_rk45.max():.2e}")
print(f"  DOP853:   {dE_dop.max():.2e}")
print(f"  Verlet:   {dE_verlet.max():.2e}")
print(f"  BM4:      {dE_bm4.max():.2e}")

print(f"\n── Max ellipse residual ─────────────────")
print(f"  Taylor:   {R_taylor.max():.2e}")
print(f"  RK45:     {R_rk45.max():.2e}")
print(f"  DOP853:   {R_dop.max():.2e}")
print(f"  Verlet:   {R_verlet.max():.2e}")
print(f"  BM4:      {R_bm4.max():.2e}")

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(hist[:,0],       hist[:,1],       lw=1.5, label=f'Taylor adapt ≤ {order}')
ax.plot(sol_rk45.y[0],   sol_rk45.y[1],  lw=1, ls='--',            label='RK45')
ax.plot(sol_dop.y[0],    sol_dop.y[1],   lw=1, ls=':',             label='DOP853')
ax.plot(sol_verlet.y[0], sol_verlet.y[1],lw=1, ls='-.',            label='Verlet')
ax.plot(sol_bm4.y[0],    sol_bm4.y[1],   lw=1, ls=(0,(3,1,1,1)),  label='BM4')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Kepler orbit (e={e})')
ax.set_aspect('equal'); ax.legend(); ax.grid(True)

ax = axes[1]
ax.semilogy(t_eval, dE_taylor+1e-20, lw=1.5, label=f'Taylor adapt ≤ {order}')
ax.semilogy(t_eval, dE_rk45  +1e-20, lw=1, ls='--',            label='RK45')
ax.semilogy(t_eval, dE_dop   +1e-20, lw=1, ls=':',             label='DOP853')
ax.semilogy(t_eval, dE_verlet+1e-20, lw=1, ls='-.',            label='Verlet')
ax.semilogy(t_eval, dE_bm4   +1e-20, lw=1, ls=(0,(3,1,1,1)),  label='BM4')
ax.set_xlabel('t'); ax.set_ylabel('|ΔE|')
ax.set_title('Energy drift'); ax.legend(); ax.grid(True)

ax = axes[2]
ax.semilogy(t_eval, R_taylor+1e-20, lw=1.5, label=f'Taylor adapt ≤ {order}')
ax.semilogy(t_eval, R_rk45  +1e-20, lw=1, ls='--',            label='RK45')
ax.semilogy(t_eval, R_dop   +1e-20, lw=1, ls=':',             label='DOP853')
ax.semilogy(t_eval, R_verlet+1e-20, lw=1, ls='-.',            label='Verlet')
ax.semilogy(t_eval, R_bm4   +1e-20, lw=1, ls=(0,(3,1,1,1)),  label='BM4')
ax.set_xlabel('t'); ax.set_ylabel('ellipse residual')
ax.set_title('Geometric orbit error'); ax.legend(); ax.grid(True)

plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t_arr[:-1], used_order, '.-')
plt.xlabel('t'); plt.ylabel('used Taylor order')
plt.title(f'Adaptive Taylor order (tol={tol_order:.0e}, e={e})')
plt.grid(True); plt.tight_layout(); plt.show()

labels = [f'Taylor\nadapt≤{order}', 'RK45', 'DOP853', 'Verlet\n(pyham)', 'BM4\n(pyham)']
times  = [time_taylor, time_rk45, time_dop, time_verlet, time_bm4]

plt.figure(figsize=(8, 4))
bars = plt.bar(labels, times, color=['steelblue','tomato','tomato','seagreen','seagreen'])
plt.bar_label(bars, fmt='%.2fs', padding=3)
plt.ylabel('Wall time (s)')
plt.title(f'Integration time  (t=[0,{t_span[1]:.2f}], h={h}, e={e})')
plt.tight_layout(); plt.show()

# ── dynamic ranking table ─────────────────────────────────────────────────────
methods = [
    {"name": f"Taylor adapt≤{order}", "time": float(time_taylor),
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

# ranking
for key, field in [("rank_time","time"),("rank_dE","dE"),("rank_res","res")]:
    for rank, idx in enumerate(sorted(range(len(methods)),
                                      key=lambda i: methods[i][field]), start=1):
        methods[idx][key] = rank

taylor_time = methods[0]["time"]
taylor_dE   = methods[0]["dE"]
taylor_res  = methods[0]["res"]

for m in methods:
    m["speedup_vs_taylor"] = taylor_time / m["time"]
    m["dE_vs_taylor"]      = m["dE"]    / taylor_dE
    m["res_vs_taylor"]     = m["res"]   / taylor_res
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
