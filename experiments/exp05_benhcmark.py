import numpy as np
import matplotlib.pyplot as plt
import math
import time
#from numba import njit, float64, int64
from numba import njit as _njit, float64, int64
def njit(*args, **kwargs):
    def decorator(fn):
        return fn
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

# ── shared parameters ─────────────────────────────────────────────────────────
ORDER     = 30
T_SPAN    = (0.0, math.pi * 5)
H         = math.pi / 100
TOL_ORDER = 1e-16
MIN_ORDER = 2
ECC       = 0.75
N_RUNS    = 5

t_steps = int((T_SPAN[1] - T_SPAN[0]) / H)
t_eval  = np.linspace(T_SPAN[0], T_SPAN[1], t_steps + 1)

BINOM = np.array([[math.comb(n, k) if k <= n else 0
                   for k in range(ORDER + 1)]
                  for n in range(ORDER + 1)], dtype=np.float64)
FACT  = np.array([math.factorial(k) for k in range(ORDER + 1)], dtype=np.float64)
HPOW  = np.array([H**k              for k in range(ORDER + 1)], dtype=np.float64)

# ── initial conditions ────────────────────────────────────────────────────────
y1_0 = 1.0 - ECC;  y2_0 = 0.0;  y3_0 = 0.0
y4_0 = math.sqrt((1.0 + ECC) / (1.0 - ECC))
r0   = math.sqrt(y1_0**2 + y2_0**2)
y5_0 = r0**3; y6_0 = r0; y7_0 = 1/r0**3; y8_0 = 1/r0

state_ode = np.array([y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0, y8_0])

y9_0  = y1_0*y7_0;  y10_0 = y2_0*y7_0
y11_0 = y1_0*y3_0;  y12_0 = y2_0*y4_0
y13_0 = y7_0*y7_0;  y14_0 = y8_0*y8_0
y15_0 = y6_0*y11_0; y16_0 = y6_0*y12_0
y17_0 = y8_0*y11_0; y18_0 = y8_0*y12_0
y19_0 = y6_0*y13_0; y20_0 = y8_0*y14_0
y21_0 = y11_0*y19_0; y22_0 = y12_0*y19_0
y23_0 = y11_0*y20_0; y24_0 = y12_0*y20_0

state_dae = np.array([
    y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0, y8_0,
    y9_0, y10_0, y11_0, y12_0, y13_0, y14_0, y15_0, y16_0,
    y17_0, y18_0, y19_0, y20_0, y21_0, y22_0, y23_0, y24_0,
])

# ── global op counters ────────────────────────────────────────────────────────
# OPS[0] = multiplications,  OPS[1] = additions
# to disable: comment out OPS[:] = 0 and the count_* calls in the runners
OPS = np.zeros(2, dtype=np.int64)

# ── JIT functions ─────────────────────────────────────────────────────────────
@njit(float64(float64[:,:], int64, int64, int64, float64[:,:]), cache=True, fastmath=False)
def prod2(TC, a, b, n, BINOM):
    row = BINOM[n]; s = 0.0
    for j in range(n + 1):
        s += row[j] * TC[j, a] * TC[n-j, b]
    return s

@njit(float64(float64[:,:], int64, int64, int64, int64, float64[:,:]), cache=True, fastmath=False)
def prod3(TC, a, b, c, n, BINOM):
    row_n = BINOM[n]; s = 0.0
    for j in range(n + 1):
        row_nj = BINOM[n - j]; tc_ja = TC[j, a]
        for m in range(n - j + 1):
            s += row_n[j] * row_nj[m] * tc_ja * TC[m, b] * TC[n-j-m, c]
    return s

@njit(float64(float64[:,:], int64, float64[:,:]), cache=True, fastmath=False)
def S_deriv(TC, n, BINOM):
    return prod2(TC, 0, 2, n, BINOM) + prod2(TC, 1, 3, n, BINOM)

@njit(cache=True, fastmath=False)
def taylor_step_ode(sv, TC, S_cache, r_invr3_sq, invr_cubed, x_invr3, y_invr3,
                    BINOM, FACT, HPOW, order, min_order, tol):
    TC[0] = sv
    S_cache[0]    = S_deriv(TC, 0, BINOM)
    r_invr3_sq[0] = prod3(TC, 5, 6, 6, 0, BINOM)
    invr_cubed[0] = prod3(TC, 7, 7, 7, 0, BINOM)
    x_invr3[0]    = prod2(TC, 0, 6, 0, BINOM)
    y_invr3[0]    = prod2(TC, 1, 6, 0, BINOM)

    new_state = sv.copy(); used_ord = order
    for k in range(1, order + 1):
        n = k - 1
        S_cache[n]    = S_deriv(TC, n, BINOM)
        r_invr3_sq[n] = prod3(TC, 5, 6, 6, n, BINOM)
        invr_cubed[n] = prod3(TC, 7, 7, 7, n, BINOM)
        x_invr3[n]    = prod2(TC, 0, 6, n, BINOM)
        y_invr3[n]    = prod2(TC, 1, 6, n, BINOM)
        nonlin = np.zeros(6)
        nonlin[0] = x_invr3[n]; nonlin[1] = y_invr3[n]
        row = BINOM[n]
        for j in range(n + 1):
            c = row[j]; Snj = S_cache[n - j]
            nonlin[2] += c * TC[j, 5]      * Snj
            nonlin[3] += c * TC[j, 7]      * Snj
            nonlin[4] += c * r_invr3_sq[j] * Snj
            nonlin[5] += c * invr_cubed[j] * Snj
        TC[k, 0] = TC[k-1, 2]; TC[k, 1] = TC[k-1, 3]
        TC[k, 2] = -nonlin[0]; TC[k, 3] = -nonlin[1]
        TC[k, 4] =  3.0*nonlin[2]; TC[k, 5] = nonlin[3]
        TC[k, 6] = -3.0*nonlin[4]; TC[k, 7] = -nonlin[5]
        term = TC[k] * HPOW[k] / FACT[k]
        new_state += term
        term_size = 0.0
        for v in range(8):
            av = abs(term[v])
            if av > term_size: term_size = av
        if k >= min_order and term_size < tol:
            used_ord = k; break
    return new_state, used_ord

@njit(cache=True, fastmath=False)
def taylor_step_dae(sv, TC, BINOM, FACT, HPOW, order, min_order, tol):
    TC[0] = sv
    new_state = sv.copy(); used_ord = order
    for k in range(1, order + 1):
        n = k - 1
        TC[n,  8] = prod2(TC, 0,  6, n, BINOM)
        TC[n,  9] = prod2(TC, 1,  6, n, BINOM)
        TC[n, 10] = prod2(TC, 0,  2, n, BINOM)
        TC[n, 11] = prod2(TC, 1,  3, n, BINOM)
        TC[n, 12] = prod2(TC, 6,  6, n, BINOM)
        TC[n, 13] = prod2(TC, 7,  7, n, BINOM)
        TC[n, 14] = prod2(TC,  5, 10, n, BINOM)
        TC[n, 15] = prod2(TC,  5, 11, n, BINOM)
        TC[n, 16] = prod2(TC,  7, 10, n, BINOM)
        TC[n, 17] = prod2(TC,  7, 11, n, BINOM)
        TC[n, 18] = prod2(TC,  5, 12, n, BINOM)
        TC[n, 19] = prod2(TC,  7, 13, n, BINOM)
        TC[n, 20] = prod2(TC, 10, 18, n, BINOM)
        TC[n, 21] = prod2(TC, 11, 18, n, BINOM)
        TC[n, 22] = prod2(TC, 10, 19, n, BINOM)
        TC[n, 23] = prod2(TC, 11, 19, n, BINOM)
        TC[k, 0] =  TC[k-1, 2];           TC[k, 1] =  TC[k-1, 3]
        TC[k, 2] = -TC[n,  8];             TC[k, 3] = -TC[n,  9]
        TC[k, 4] =  3.0*TC[n,14]+3.0*TC[n,15]
        TC[k, 5] =      TC[n,16]+    TC[n,17]
        TC[k, 6] = -3.0*TC[n,20]-3.0*TC[n,21]
        TC[k, 7] =     -TC[n,22]-    TC[n,23]
        term = TC[k] * HPOW[k] / FACT[k]
        new_state += term
        term_size = 0.0
        for v in range(8):
            av = abs(term[v])
            if av > term_size: term_size = av
        if k >= min_order and term_size < tol:
            used_ord = k; break
    return new_state, used_ord

# ── op counting (mirrors loop structure of JIT functions) ─────────────────────
# prod2 at level n:  loop j=0..n → 2 muls + 1 add per iteration
def count_prod2(n):
    OPS[0] += 2 * (n + 1)
    OPS[1] +=     (n + 1)

# prod3 at level n:  double loop → (n+1)(n+2)/2 terms × (2 muls + 1 add)
def count_prod3(n):
    terms = (n + 1) * (n + 2) // 2
    OPS[0] += 2 * terms
    OPS[1] +=     terms

# full ODE level n
def count_ode_level(n):
    count_prod2(n); count_prod2(n); OPS[1] += 1   # S_deriv = 2×prod2 + 1 add
    count_prod3(n)                                 # r_invr3_sq
    count_prod3(n)                                 # invr_cubed
    count_prod2(n)                                 # x_invr3
    count_prod2(n)                                 # y_invr3
    OPS[0] += 8 * (n + 1)                         # nonlin[2..5] j-loop: 4×2 muls
    OPS[1] += 4 * (n + 1)                         # nonlin[2..5] j-loop: 4 adds
    OPS[0] += 4                                    # TC[k] scalar muls (3×, -3×)
    OPS[1] += 5                                    # TC[k] adds (y5',y6',y7',y8')
    OPS[0] += 8; OPS[1] += 8                      # term accumulation (8 vars)

# full DAE level n
def count_dae_level(n):
    for _ in range(16): count_prod2(n)             # 16 algebraic prod2 calls
    OPS[0] += 4; OPS[1] += 4                      # TC[k] ODE step
    OPS[0] += 24; OPS[1] += 24                    # term accumulation (24 vars)

# ── work arrays ───────────────────────────────────────────────────────────────
TC_ode     = np.empty((ORDER + 1, 8),  dtype=float)
S_cache    = np.empty(ORDER, dtype=float)
r_invr3_sq = np.empty(ORDER, dtype=float)
invr_cubed = np.empty(ORDER, dtype=float)
x_invr3    = np.empty(ORDER, dtype=float)
y_invr3    = np.empty(ORDER, dtype=float)
TC_dae     = np.empty((ORDER + 1, 24), dtype=float)

# ── warmup ────────────────────────────────────────────────────────────────────
print("Warmup ODE...", end=" ", flush=True)
taylor_step_ode(state_ode.copy(), TC_ode, S_cache, r_invr3_sq, invr_cubed,
                x_invr3, y_invr3, BINOM, FACT, HPOW, ORDER, MIN_ORDER, TOL_ORDER)
print("done")
print("Warmup DAE...", end=" ", flush=True)
taylor_step_dae(state_dae.copy(), TC_dae, BINOM, FACT, HPOW, ORDER, MIN_ORDER, TOL_ORDER)
print("done")

# ── runners ───────────────────────────────────────────────────────────────────
def run_ode(count_ops=False):
    sv = state_ode.copy()
    hist = np.zeros((t_steps + 1, 8)); hist[0] = sv
    ord_hist = np.zeros(t_steps, dtype=int)
    if count_ops: OPS[:] = 0
    for i in range(t_steps):
        sv, uo = taylor_step_ode(sv, TC_ode, S_cache, r_invr3_sq, invr_cubed,
                                 x_invr3, y_invr3, BINOM, FACT, HPOW,
                                 ORDER, MIN_ORDER, TOL_ORDER)
        if count_ops:
            for k in range(1, uo + 1):
                count_ode_level(k - 1)
        hist[i+1] = sv; ord_hist[i] = uo
    return hist, ord_hist

def run_dae(count_ops=False):
    sv = state_dae.copy()
    hist = np.zeros((t_steps + 1, 24)); hist[0] = sv
    ord_hist = np.zeros(t_steps, dtype=int)
    if count_ops: OPS[:] = 0
    for i in range(t_steps):
        sv, uo = taylor_step_dae(sv, TC_dae, BINOM, FACT, HPOW,
                                 ORDER, MIN_ORDER, TOL_ORDER)
        if count_ops:
            for k in range(1, uo + 1):
                count_dae_level(k - 1)
        hist[i+1] = sv; ord_hist[i] = uo
    return hist, ord_hist

# ── timing ────────────────────────────────────────────────────────────────────
print(f"\nBenchmarking {N_RUNS} runs each...")
times_ode = []
for r in range(N_RUNS):
    t0 = time.perf_counter()
    hist_ode, ord_ode = run_ode()
    times_ode.append(time.perf_counter() - t0)
    print(f"  ODE run {r+1}: {times_ode[-1]:.4f}s")

times_dae = []
for r in range(N_RUNS):
    t0 = time.perf_counter()
    hist_dae, ord_dae = run_dae()
    times_dae.append(time.perf_counter() - t0)
    print(f"  DAE run {r+1}: {times_dae[-1]:.4f}s")

time_ode = sorted(times_ode)[N_RUNS // 2]
time_dae = sorted(times_dae)[N_RUNS // 2]

# ── op count (single pass, no timing pressure) ────────────────────────────────
print("\nCounting operations...")
run_ode(count_ops=True)
ode_muls, ode_adds = int(OPS[0]), int(OPS[1])

run_dae(count_ops=True)
dae_muls, dae_adds = int(OPS[0]), int(OPS[1])

# ── diagnostics ───────────────────────────────────────────────────────────────
def energy(x, y, vx, vy):
    return 0.5*(vx**2 + vy**2) - 1.0/np.sqrt(x**2 + y**2)

def ellipse_res(x, y, e):
    return np.abs((x + e)**2 + y**2/(1.0 - e**2) - 1.0)

E_ode = energy(hist_ode[:,0], hist_ode[:,1], hist_ode[:,2], hist_ode[:,3])
E_dae = energy(hist_dae[:,0], hist_dae[:,1], hist_dae[:,2], hist_dae[:,3])
dE_ode = np.abs(E_ode - E_ode[0])
dE_dae = np.abs(E_dae - E_dae[0])
R_ode = ellipse_res(hist_ode[:,0], hist_ode[:,1], ECC)
R_dae = ellipse_res(hist_dae[:,0], hist_dae[:,1], ECC)

us_ode = time_ode / t_steps * 1e6
us_dae = time_dae / t_steps * 1e6

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*76}")
print(f"  order={ORDER}, h=π/{round(math.pi/H)}, e={ECC},  median/{N_RUNS} runs,  {t_steps} steps total")
print(f"{'─'*76}")
print(f"  {'':12}  {'vars':>5}  {'time(s)':>8}  {'µs/step':>8}  "
      f"{'muls':>10}  {'adds':>10}  {'total ops':>12}  {'avg ord':>8}")
print(f"  {'─'*74}")
print(f"  {'ODE':<12}  {'8':>5}  {time_ode:>8.4f}  {us_ode:>8.1f}  "
      f"{ode_muls:>10,}  {ode_adds:>10,}  {ode_muls+ode_adds:>12,}  {ord_ode.mean():>8.2f}")
print(f"  {'DAE':<12}  {'24':>5}  {time_dae:>8.4f}  {us_dae:>8.1f}  "
      f"{dae_muls:>10,}  {dae_adds:>10,}  {dae_muls+dae_adds:>12,}  {ord_dae.mean():>8.2f}")
print(f"{'─'*76}")
print(f"  DAE/ODE:  time={time_dae/time_ode:.2f}x   ops={(dae_muls+dae_adds)/(ode_muls+ode_adds):.2f}x")
print(f"  ops/µs:   ODE={(ode_muls+ode_adds)/(time_ode*1e6):.0f}   DAE={(dae_muls+dae_adds)/(time_dae*1e6):.0f}")

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f'ODE vs DAE — order={ORDER}, h=π/{round(math.pi/H)}, e={ECC}')

ax = axes[0]
bars = ax.bar(['ODE\n(8 vars)', 'DAE\n(24 vars)'],
              [time_ode, time_dae], color=['steelblue', 'tomato'])
ax.bar_label(bars, fmt='%.4fs', padding=3)
ax.set_ylabel('Wall time (s)'); ax.set_title('Time (median)'); ax.grid(True, axis='y')

ax = axes[1]
width = 0.35; x = np.arange(2)
b1 = ax.bar(x - width/2, [ode_muls, dae_muls], width, label='muls',
            color=['#4a90d9','#4a90d9'])
b2 = ax.bar(x + width/2, [ode_adds, dae_adds], width, label='adds',
            color=['#7fbfef','#7fbfef'])
ax.bar_label(b1, fmt='%d', padding=2, fontsize=8)
ax.bar_label(b2, fmt='%d', padding=2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(['ODE\n(8 vars)', 'DAE\n(24 vars)'])
ax.set_ylabel('operations (full run)'); ax.set_title('Op count')
ax.legend(); ax.grid(True, axis='y')

ax = axes[2]
ax.semilogy(t_eval, dE_ode+1e-20, lw=2,   label='ODE', color='steelblue')
ax.semilogy(t_eval, dE_dae+1e-20, lw=1.5, label='DAE', color='tomato', ls='--')
ax.set_xlabel('t'); ax.set_ylabel('|ΔE|')
ax.set_title('Energy drift'); ax.legend(); ax.grid(True)

plt.tight_layout(); plt.show()