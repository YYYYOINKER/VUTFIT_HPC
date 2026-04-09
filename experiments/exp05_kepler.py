import numpy as np;
import matplotlib.pyplot as plt;
import math;
import time;
import sympy as sp;
from scipy.integrate import solve_ivp;
from pyhamsys import HamSys, solve_ivp_sympext, Parameters;

# OPTIMIZATIONS
# - precompute binomial coefficients instead of using math.comb(...) directly in loop
# - precompute factorials instead of using math.factorial(k) directly in loop
# - precompute h**k
# - momoization of nonlinear taylor terms

# ── parameters ────────────────────────────────────────────────────────────────
order      = 20;
t_span     = (0.0, np.pi * 32);
h          = 0.01;
tol_order  = 1e-12;
min_order  = 2;
e          = 0.75;   # eccentricity; use e=0.0 for circular orbit

t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / h) + 1);

# ── precompute combinatorics / Taylor weights ────────────────────────────────
BINOM = [[math.comb(n, k) for k in range(n + 1)] for n in range(order + 1)];
FACT  = [math.factorial(k) for k in range(order + 1)];
HPOW  = [h**k for k in range(order + 1)];

# ── initial conditions (general Kepler ellipse at periapsis) ────────────────
# x0 = 1 - e, y0 = 0, vx0 = 0, vy0 = sqrt((1+e)/(1-e))
y1_0 = 1.0 - e;
y2_0 = 0.0;
y3_0 = 0.0;
y4_0 = math.sqrt((1.0 + e) / (1.0 - e));

r0   = math.sqrt(y1_0**2 + y2_0**2);

# [x, y, vx, vy, r^3, r, 1/r^3, 1/r]
y5_0 = r0**3;
y6_0 = r0;
y7_0 = 1.0 / y5_0;   # 1 / r^3
y8_0 = 1.0 / y6_0;   # 1 / r

state_vector = np.array([y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0, y8_0], dtype=float);

t_steps = int((t_span[1] - t_span[0]) / h);
t_arr   = np.zeros(t_steps + 1);
hist    = np.zeros((t_steps + 1, 8));
used_order = np.zeros(t_steps, dtype=int);

t_arr[0] = t_span[0];
hist[0]  = state_vector;

# ── matrix setup ──────────────────────────────────────────────────────────────
A = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float);

# right-hand nonlinear vector will be:
# [x*(1/r^3), y*(1/r^3), r*S, (1/r)*S, r*(1/r^3)^2*S, (1/r)^3*S]
# and B routes them into equations:
# vx' = -x/r^3
# vy' = -y/r^3
# (r^3)' = 3 r S
# r'     = (1/r) S
# (1/r^3)' = -3 r (1/r^3)^2 S
# (1/r)'   = -(1/r)^3 S
B = np.array([
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0],
    [ 0, -1,  0,  0,  0,  0],
    [ 0,  0,  3,  0,  0,  0],
    [ 0,  0,  0,  1,  0,  0],
    [ 0,  0,  0,  0, -3,  0],
    [ 0,  0,  0,  0,  0, -1],
], dtype=float);

# ── helpers ───────────────────────────────────────────────────────────────────
def prod2(derivs, a, b, n):
    s = 0.0;
    row = BINOM[n];
    for j in range(n + 1):
        s += row[j] * derivs[j, a] * derivs[n - j, b];
    return s;

def prod3(derivs, a, b, c, n):
    s = 0.0;
    row_n = BINOM[n];
    for j in range(n + 1):
        row_nj = BINOM[n - j];
        dj_a = derivs[j, a];
        for m in range(n - j + 1):
            l = n - j - m;
            s += row_n[j] * row_nj[m] * dj_a * derivs[m, b] * derivs[l, c];
    return s;

def S_deriv(derivs, n):
    return prod2(derivs, 0, 2, n) + prod2(derivs, 1, 3, n);

# ── Taylor integration loop with adaptive order ──────────────────────────────
t0_taylor = time.perf_counter();

derivs = np.empty((order + 1, 8), dtype=float);
Svals  = np.empty(order, dtype=float);
P766   = np.empty(order, dtype=float);   # r*(1/r^3)^2
P888   = np.empty(order, dtype=float);   # (1/r)^3

for i in range(t_steps):
    derivs[0] = state_vector;
    reached_order = order;

    for k in range(1, order + 1):
        n = k - 1;
        y_jk_k = np.zeros(6, dtype=float);

        for m in range(n + 1):
            Svals[m] = S_deriv(derivs, m);
            P766[m]  = prod3(derivs, 5, 6, 6, m);  # r * (1/r^3)^2
            P888[m]  = prod3(derivs, 7, 7, 7, m);  # (1/r)^3

        # x/r^3, y/r^3
        y_jk_k[0] = prod2(derivs, 0, 6, n);
        y_jk_k[1] = prod2(derivs, 1, 6, n);

        row = BINOM[n];
        for j in range(n + 1):
            c   = row[j];
            Snj = Svals[n - j];

            # r*S
            y_jk_k[2] += c * derivs[j, 5] * Snj;

            # (1/r)*S
            y_jk_k[3] += c * derivs[j, 7] * Snj;

            # r*(1/r^3)^2*S
            y_jk_k[4] += c * P766[j] * Snj;

            # (1/r)^3*S
            y_jk_k[5] += c * P888[j] * Snj;

        derivs[k] = A @ derivs[k - 1] + B @ y_jk_k;

        term_k = derivs[k] * HPOW[k] / FACT[k];
        term_norm = np.max(np.abs(term_k));

        if k >= min_order and term_norm < tol_order:
            reached_order = k;
            break;

    new_state = state_vector.copy();
    for k in range(1, reached_order + 1):
        new_state += derivs[k] * HPOW[k] / FACT[k];

    state_vector = new_state;
    used_order[i] = reached_order;

    t_arr[i + 1] = t_arr[i] + h;
    hist[i + 1]  = state_vector;

t1_taylor = time.perf_counter();
time_taylor = t1_taylor - t0_taylor;

# ── scipy RK45 / DOP853 ───────────────────────────────────────────────────────
def kepler_rhs(t, y):
    x, y_pos, vx, vy = y;
    r3 = (x * x + y_pos * y_pos)**1.5;
    return [vx, vy, -x / r3, -y_pos / r3];

y0_phys = [y1_0, y2_0, y3_0, y4_0];

t0_rk45 = time.perf_counter();
sol_rk45 = solve_ivp(
    kepler_rhs,
    t_span,
    y0_phys,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
);
t1_rk45 = time.perf_counter();
time_rk45 = t1_rk45 - t0_rk45;

t0_dop = time.perf_counter();
sol_dop = solve_ivp(
    kepler_rhs,
    t_span,
    y0_phys,
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
);
t1_dop = time.perf_counter();
time_dop = t1_dop - t0_dop;

# ── pyhamsys: Verlet (order 2) and BM4 (order 4) ─────────────────────────────
hs = HamSys(ndof=2);
q1s, q2s, p1s, p2s = sp.symbols('q1 q2 p1 p2');
H_kepler = lambda q1, q2, p1, p2, t: (p1**2 + p2**2) / 2 - 1 / sp.sqrt(q1**2 + q2**2);
hs.compute_vector_field(H_kepler);

y0_ham = np.array([y1_0, y2_0, y3_0, y4_0]);

t0_verlet = time.perf_counter();
sol_verlet = solve_ivp_sympext(
    hs,
    t_span,
    y0_ham,
    Parameters(step=h, solver='Verlet'),
    t_eval=t_eval,
);
t1_verlet = time.perf_counter();
time_verlet = t1_verlet - t0_verlet;

t0_bm4 = time.perf_counter();
sol_bm4 = solve_ivp_sympext(
    hs,
    t_span,
    y0_ham,
    Parameters(step=h, solver='BM4'),
    t_eval=t_eval,
);
t1_bm4 = time.perf_counter();
time_bm4 = t1_bm4 - t0_bm4;

# ── diagnostics ───────────────────────────────────────────────────────────────
def energy(x, y, vx, vy):
    r = np.sqrt(x**2 + y**2);
    return 0.5 * (vx**2 + vy**2) - 1.0 / r;

def ellipse_residual(x, y, e):
    if abs(e) < 1e-15:
        return np.abs(x**2 + y**2 - 1.0);
    return np.abs((x + e)**2 + y**2 / (1.0 - e**2) - 1.0);

E_taylor = energy(hist[:, 0], hist[:, 1], hist[:, 2], hist[:, 3]);
E_rk45   = energy(sol_rk45.y[0], sol_rk45.y[1], sol_rk45.y[2], sol_rk45.y[3]);
E_dop    = energy(sol_dop.y[0],  sol_dop.y[1],  sol_dop.y[2],  sol_dop.y[3]);
E_verlet = energy(sol_verlet.y[0], sol_verlet.y[1], sol_verlet.y[2], sol_verlet.y[3]);
E_bm4    = energy(sol_bm4.y[0],   sol_bm4.y[1],   sol_bm4.y[2],   sol_bm4.y[3]);

dE_taylor = np.abs(E_taylor - E_taylor[0]);
dE_rk45   = np.abs(E_rk45  - E_rk45[0]);
dE_dop    = np.abs(E_dop   - E_dop[0]);
dE_verlet = np.abs(E_verlet - E_verlet[0]);
dE_bm4    = np.abs(E_bm4   - E_bm4[0]);

R_taylor = ellipse_residual(hist[:, 0], hist[:, 1], e);
R_rk45   = ellipse_residual(sol_rk45.y[0], sol_rk45.y[1], e);
R_dop    = ellipse_residual(sol_dop.y[0],  sol_dop.y[1], e);
R_verlet = ellipse_residual(sol_verlet.y[0], sol_verlet.y[1], e);
R_bm4    = ellipse_residual(sol_bm4.y[0],   sol_bm4.y[1], e);

# ── summary prints ────────────────────────────────────────────────────────────
'''print("\n── Taylor adaptive-order settings ─────────────────");
print(f"  max order      = {order}");
print(f"  min order      = {min_order}");
print(f"  order tol      = {tol_order:.1e}");
print(f"  eccentricity   = {e}");
print(f"  avg used order = {used_order.mean():.2f}");
print(f"  min used order = {used_order.min()}");
print(f"  max used order = {used_order.max()}");

print("\n── Timing ───────────────────────────────");
print(f"  Taylor adaptive (max ord {order}):  {time_taylor:.3f} s");
print(f"  scipy RK45:                         {time_rk45:.3f} s");
print(f"  scipy DOP853:                       {time_dop:.3f} s");
print(f"  pyhamsys Verlet:                    {time_verlet:.3f} s");
print(f"  pyhamsys BM4:                       {time_bm4:.3f} s");

print("\n── Max |ΔE| ─────────────────────────────");
print(f"  Taylor:   {dE_taylor.max():.2e}");
print(f"  RK45:     {dE_rk45.max():.2e}");
print(f"  DOP853:   {dE_dop.max():.2e}");
print(f"  Verlet:   {dE_verlet.max():.2e}");
print(f"  BM4:      {dE_bm4.max():.2e}");

print("\n── Max ellipse residual ─────────────────");
print(f"  Taylor:   {R_taylor.max():.2e}");
print(f"  RK45:     {R_rk45.max():.2e}");
print(f"  DOP853:   {R_dop.max():.2e}");
print(f"  Verlet:   {R_verlet.max():.2e}");
print(f"  BM4:      {R_bm4.max():.2e}");'''

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5));

ax = axes[0];
ax.plot(hist[:, 0],       hist[:, 1],       lw=1.5, label=f'Taylor adapt ≤ {order}');
ax.plot(sol_rk45.y[0],    sol_rk45.y[1],    lw=1, ls='--', label='RK45');
ax.plot(sol_dop.y[0],     sol_dop.y[1],     lw=1, ls=':',  label='DOP853');
ax.plot(sol_verlet.y[0],  sol_verlet.y[1],  lw=1, ls='-.', label='Verlet');
ax.plot(sol_bm4.y[0],     sol_bm4.y[1],     lw=1, ls=(0, (3, 1, 1, 1)), label='BM4');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_title(f'Kepler orbit (e={e})');
ax.set_aspect('equal');
ax.legend();
ax.grid(True);

ax = axes[1];
ax.semilogy(t_eval, dE_taylor + 1e-20, lw=1.5, label=f'Taylor adapt ≤ {order}');
ax.semilogy(t_eval, dE_rk45   + 1e-20, lw=1, ls='--', label='RK45');
ax.semilogy(t_eval, dE_dop    + 1e-20, lw=1, ls=':',  label='DOP853');
ax.semilogy(t_eval, dE_verlet + 1e-20, lw=1, ls='-.', label='Verlet');
ax.semilogy(t_eval, dE_bm4    + 1e-20, lw=1, ls=(0, (3, 1, 1, 1)), label='BM4');
ax.set_xlabel('t');
ax.set_ylabel('|ΔE|');
ax.set_title('Energy drift');
ax.legend();
ax.grid(True);

ax = axes[2];
ax.semilogy(t_eval, R_taylor + 1e-20, lw=1.5, label=f'Taylor adapt ≤ {order}');
ax.semilogy(t_eval, R_rk45   + 1e-20, lw=1, ls='--', label='RK45');
ax.semilogy(t_eval, R_dop    + 1e-20, lw=1, ls=':',  label='DOP853');
ax.semilogy(t_eval, R_verlet + 1e-20, lw=1, ls='-.', label='Verlet');
ax.semilogy(t_eval, R_bm4    + 1e-20, lw=1, ls=(0, (3, 1, 1, 1)), label='BM4');
ax.set_xlabel('t');
ax.set_ylabel('ellipse residual');
ax.set_title('Geometric orbit error');
ax.legend();
ax.grid(True);

plt.tight_layout();
plt.show();

# ── adaptive-order plot ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 4));
plt.plot(t_arr[:-1], used_order, '.-');
plt.xlabel('t');
plt.ylabel('used Taylor order');
plt.title(f'Adaptive Taylor order (tol={tol_order:.0e}, e={e})');
plt.grid(True);
plt.tight_layout();
plt.show();

# ── timing bar chart ──────────────────────────────────────────────────────────
labels = [f'Taylor\nadapt≤{order}', 'RK45', 'DOP853', 'Verlet\n(pyham)', 'BM4\n(pyham)'];
times  = [time_taylor, time_rk45, time_dop, time_verlet, time_bm4];

plt.figure(figsize=(8, 4));
bars = plt.bar(labels, times, color=['steelblue', 'tomato', 'tomato', 'seagreen', 'seagreen']);
plt.bar_label(bars, fmt='%.2fs', padding=3);
plt.ylabel('Wall time (s)');
plt.title(f'Integration time  (t=[0,{t_span[1]}], h={h}, e={e})');
plt.tight_layout();
plt.show();

# ── dynamic ranking table ─────────────────────────────────────────────────────
methods = [
    {
        "name": f"Taylor adapt≤{order}",
        "time": float(time_taylor),
        "dE": float(dE_taylor.max()),
        "res": float(R_taylor.max()),
    },
    {
        "name": "RK45",
        "time": float(time_rk45),
        "dE": float(dE_rk45.max()),
        "res": float(R_rk45.max()),
    },
    {
        "name": "DOP853",
        "time": float(time_dop),
        "dE": float(dE_dop.max()),
        "res": float(R_dop.max()),
    },
    {
        "name": "Verlet",
        "time": float(time_verlet),
        "dE": float(dE_verlet.max()),
        "res": float(R_verlet.max()),
    },
    {
        "name": "BM4",
        "time": float(time_bm4),
        "dE": float(dE_bm4.max()),
        "res": float(R_bm4.max()),
    },
];

time_order = sorted(range(len(methods)), key=lambda i: methods[i]["time"]);
dE_order   = sorted(range(len(methods)), key=lambda i: methods[i]["dE"]);
res_order  = sorted(range(len(methods)), key=lambda i: methods[i]["res"]);

for rank, idx in enumerate(time_order, start=1):
    methods[idx]["rank_time"] = rank;

for rank, idx in enumerate(dE_order, start=1):
    methods[idx]["rank_dE"] = rank;

for rank, idx in enumerate(res_order, start=1):
    methods[idx]["rank_res"] = rank;

taylor_time = methods[0]["time"];
taylor_dE   = methods[0]["dE"];
taylor_res  = methods[0]["res"];

for m in methods:
    m["speedup_vs_taylor"] = taylor_time / m["time"];
    m["dE_vs_taylor"]      = m["dE"] / taylor_dE;
    m["res_vs_taylor"]     = m["res"] / taylor_res;

for m in methods:
    m["score"] = m["rank_time"] + m["rank_dE"] + m["rank_res"];

overall_order = sorted(
    range(len(methods)),
    key=lambda i: (
        methods[i]["score"],
        methods[i]["rank_res"],
        methods[i]["rank_dE"],
        methods[i]["rank_time"],
    ),
);

for rank, idx in enumerate(overall_order, start=1):
    methods[idx]["rank_overall"] = rank;

best_time = min(m["time"] for m in methods);
best_dE   = min(m["dE"] for m in methods);
best_res  = min(m["res"] for m in methods);

print("\n── Dynamic ranking table ───────────────────────────────────────────────────────────────");
header = (
    f"{'Method':<18}"
    f"{'time [s]':>12}"
    f"{'max |ΔE|':>14}"
    f"{'max ellipse':>16}"
    f"{'spd/Tay':>10}"
    f"{'dE/Tay':>10}"
    f"{'res/Tay':>10}"
    f"{'r_t':>6}"
    f"{'r_E':>6}"
    f"{'r_R':>6}"
    f"{'score':>8}"
    f"{'ovr':>6}"
);
print(header);
print("-" * len(header));

for m in sorted(methods, key=lambda x: x["rank_overall"]):
    tag_time = "*" if abs(m["time"] - best_time) <= 1e-15 else "";
    tag_dE   = "*" if abs(m["dE"]   - best_dE)   <= 1e-30 else "";
    tag_res  = "*" if abs(m["res"]  - best_res)  <= 1e-30 else "";

    print(
        f"{m['name']:<18}"
        f"{m['time']:>12.4f}{tag_time:<1}"
        f"{m['dE']:>14.2e}{tag_dE:<1}"
        f"{m['res']:>16.2e}{tag_res:<1}"
        f"{m['speedup_vs_taylor']:>10.2f}"
        f"{m['dE_vs_taylor']:>10.2f}"
        f"{m['res_vs_taylor']:>10.2f}"
        f"{m['rank_time']:>6}"
        f"{m['rank_dE']:>6}"
        f"{m['rank_res']:>6}"
        f"{m['score']:>8}"
        f"{m['rank_overall']:>6}"
    );

print("\n* = best in column");

print("\n── Ranking by time ─────────────────────");
for m in sorted(methods, key=lambda x: x["rank_time"]):
    print(f"{m['rank_time']}. {m['name']:<18}  {m['time']:.4f} s");

print("\n── Ranking by energy drift ─────────────");
for m in sorted(methods, key=lambda x: x["rank_dE"]):
    print(f"{m['rank_dE']}. {m['name']:<18}  {m['dE']:.2e}");

print("\n── Ranking by ellipse residual ─────────");
for m in sorted(methods, key=lambda x: x["rank_res"]):
    print(f"{m['rank_res']}. {m['name']:<18}  {m['res']:.2e}");
