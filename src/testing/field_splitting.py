import numpy as np;
import matplotlib.pyplot as plt;

# H = T + V

def T(p):
    return 0.8 * p ** 2;

def V(q):
    return 0.1 * q ** 3;


def dT_dp(p):
    return p;

def dV_dq(q):
    return q;


# Vector fields A, B

def A_field(q, p):

    qdot = dT_dp(p);
    pdot = 0.0 * q;
    return qdot, pdot;

def B_field(q, p):
    
    qdot = 0.0 * p;
    pdot = - dV_dq(q);
    return qdot, pdot;

def H_fieldd(q, p):

    # individual fields
    aq, ap = A_field(q, p);
    bq, bp = B_field(q, p);

    return aq + bq, ap + bp;

print("Test T(2) =", T(2.0))        # 2.0
print("Test dT_dp(2) =", dT_dp(2.0))# 2.0
print("Test V(3) =", V(3.0))        # 4.5
print("Test dV_dq(3) =", dV_dq(3.0))# 3.0


q0, p0 = 1.0, 0.5
print("A_field(1,0.5) =", A_field(q0, p0))
print("B_field(1,0.5) =", B_field(q0, p0))
print("H_field(1,0.5) =", H_fieldd(q0, p0))

# field A
qmin, qmax = -2.0, 2.0;
pmin, pmax = -2.0, 2.0;
N = 21;

qg = np.linspace(qmin, qmax, N);
pg = np.linspace(pmin, pmax, N);
Q, P = np.meshgrid(qg, pg);

AQ, AP = A_field(Q, P);

plt.figure(figsize=(6,6))
plt.quiver(Q, P, AQ, AP, angles='xy', scale_units='xy', scale=None, width=0.003)
plt.title("A = L_T (drift)")
plt.xlabel("q"); plt.ylabel("p")
plt.xlim(qmin, qmax); plt.ylim(pmin, pmax)
plt.grid(True)
plt.show()


# field B
BQ, BP = B_field(Q, P)

plt.figure(figsize=(6,6))
plt.quiver(Q, P, BQ, BP, angles='xy', scale_units='xy', scale=None, width=0.003)
plt.title("B = L_V (kick)")
plt.xlabel("q"); plt.ylabel("p")
plt.xlim(qmin, qmax); plt.ylim(pmin, pmax)
plt.grid(True)
plt.show()

# Field A + B
HQ, HP = H_fieldd(Q, P)

plt.figure(figsize=(6,6))
plt.quiver(Q, P, HQ, HP, angles='xy', scale_units='xy', scale=None, width=0.003)
plt.title("H = A + B")
plt.xlabel("q"); plt.ylabel("p")
plt.xlim(qmin, qmax); plt.ylim(pmin, pmax)
plt.grid(True)
plt.show()


# End of file