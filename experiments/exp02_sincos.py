#
#   Experiment 01: Exponential growth/decay
#   Author: Pavol Mihalik, VUT FIT
#
#   Compare: Euler, RK2, RK4, specialized Taylor vs exact solution
#   set OPENBLAS_NUM_THREADS=16


import os
import time
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

import numpy as np;
import matplotlib.pyplot as plt;

# Import system
from src.systems import sin_cos;
from src.systems.sin_cos import euler;
from src.systems.sin_cos import rk4;
from src.systems.sin_cos import taylor_recursive_diff;
from src.systems.sin_cos import taylor_recursive_diff_matrix;
from src.systems.sin_cos import taylor_recursive_diff_matrix_jit, lte_over_time;

# TODO matarix calcualtions

def main():


    # Problem setup
    omega: float = 1.0;
    y0: float = 0.0;
    z0: float = 1.0;
    t_span: tuple[float, float] = (0.0, 50.0);

    # Step sizes
    h_euler: float = 0.01;
    h_rk4: float = 0.5;
    h_taylor: float = 0.1;

    # Euler
    t_eu, y_eu, z_eu = euler(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_euler, omega);

    # RK4
    t_rk4, y_rk4, z_rk4 = rk4(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_rk4, omega);

    # Taylor
    start_t = time.time();
    t_ty, y_ty, z_ty = taylor_recursive_diff(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_taylor, omega, order=10);
    end_t = time.time();
    taylor_duration = end_t - start_t;
    
    # Taylor-Matrix
    start_t_matrix = time.time();
    t_ty_m, y_ty_m, z_ty_m = taylor_recursive_diff_matrix(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_taylor, omega, order=10);
    end_t_matrix = time.time();
    matrix_taylor_duration = end_t_matrix -start_t_matrix;

    # Warm up JIT
    _ = taylor_recursive_diff_matrix_jit(t_span[0], t_span[1], y0, z0, h_taylor, omega, order=10)


    # Taylor-Matrix (JIT)
    start_t_matrix_jit = time.time();
    t_ty_m, y_ty_m, z_ty_m = taylor_recursive_diff_matrix_jit(
    t_span[0], t_span[1], y0, z0, h_taylor, omega, order=10);
    end_t_matrix_jit = time.time();
    matrix_taylor_duration_jit = end_t_matrix_jit -start_t_matrix_jit;

    # Exact solution at final time
    t_final = t_span[1];
    y_exact_final = -z0 * np.sin(omega * t_final);
    z_exact_final = z0 * np.cos(omega * t_final);

    # Errors in y at final time
    err_eu = abs(y_eu[-1] - y_exact_final);
    err_rk4 = abs(y_rk4[-1] - y_exact_final);
    err_ty  = abs(y_ty[-1]  - y_exact_final);
    err_ty_m  = abs(y_ty_m[-1]  - y_exact_final);

    # Print results
    print(f"\nFinal time t = {t_final}, exact y = {y_exact_final:.6e}");
    print(f"{'Method':<15} | {'h':>8} | {'Final Error':>15}");
    print("-" * 45);
    print(f"{'Euler':<15} | {h_euler:8.3f} | {err_eu:15.6e}");
    print(f"{'RK4':<15} | {h_rk4:8.3f} | {err_rk4:15.6e}");
    print(f"{'Taylor n=10':<15} | {h_taylor:8.3f} | {err_ty:15.6e}");
    print(f'Duration: Matrix:{matrix_taylor_duration} vs Classic:{taylor_duration} vs JIT:{matrix_taylor_duration_jit}');

    # Plot solutions

    # Euler
    plt.figure(figsize=(8, 5));
    plt.plot(t_eu, y_eu, "o-", label=f"Euler (h={h_euler})");
    plt.plot(t_eu, z_eu, "o-", label=f"Euler (h={h_euler})");

    # Runge-Kutta
    plt.plot(t_rk4, y_rk4, "^-", label=f"RK4 (h={h_rk4})");
    plt.plot(t_rk4, z_rk4, "^-", label=f"RK4 (h={h_rk4})");

    # Taylor
    plt.plot(t_ty, y_ty, "d-", label=f"Taylor n=10 (h={h_taylor})");
    plt.plot(t_ty, z_ty, "d-", label=f"Taylor n=10 (h={h_taylor})");

    # Taylor Matrix
    plt.plot(t_ty_m, y_ty_m, "d-", label=f"Matrix Taylor n=10 (h={h_taylor})");
    plt.plot(t_ty_m, z_ty_m, "d-", label=f"Matrix Taylor n=10 (h={h_taylor})");

    # Exact curve
    t_dense = np.linspace(*t_span, 500);
    y_exact = -z0 * np.sin(omega * t_dense);
    z_exact = z0 * np.cos(omega * t_dense);
    
    # Exact solutions
    plt.plot(t_dense, y_exact, "k--", label="Exact solution y");
    plt.plot(t_dense, z_exact, "k--", label="Exact solution z");

    # Details
    plt.xlabel("t");
    plt.ylabel("y(t), z(t)");
    plt.grid();
    plt.legend();
    plt.title("Experiment 01: Harmonic oscillator system");

    # Save input data and output data to a text file
    with open('results/exp02/data.txt', 'a') as file:

        file.write(f"Problem: y' = -ωz, z' = ωy ; y(0)={y0}, z(0)={z0}, ω={omega}\n");
        file.write(f"time span: <{t_span[0]};{t_span[1]}>\n");
        file.write(f"{'Method':<15} | {'h':>8} | {'Final Error':>15}\n");
        file.write("-" * 45 + "\n");
        file.write(f"{'Euler':<15} | {h_euler:8.3f} | {err_eu:15.6e}\n");
        file.write(f"{'RK4':<15} | {h_rk4:8.3f} | {err_rk4:15.6e}\n");
        file.write(f"{'Taylor n=10':<15} | {h_taylor:8.3f} | {err_ty:15.6e}\n\n");

    plt.savefig('results/exp02/experiment02.png');
    plt.show();

    # exact arrays on the grids you actually used
    def exact_yz(t, omega, y0=0.0, z0=1.0):
        return np.vstack((-z0*np.sin(omega*t), z0*np.cos(omega*t)));  # shape (2, len(t))

    # errors
    Yex_eu = exact_yz(t_eu, omega);  err_eu_t = np.linalg.norm(np.vstack((y_eu, z_eu)) - Yex_eu, axis=0);
    Yex_rk4 = exact_yz(t_rk4, omega); err_rk4_t = np.linalg.norm(np.vstack((y_rk4, z_rk4)) - Yex_rk4, axis=0);
    Yex_ty = exact_yz(t_ty, omega);   err_ty_t = np.linalg.norm(np.vstack((y_ty,  z_ty )) - Yex_ty, axis=0);
    Yex_tym = exact_yz(t_ty_m, omega);err_tym_t= np.linalg.norm(np.vstack((y_ty_m,z_ty_m)) - Yex_tym, axis=0);

    # final-time errors
    err_eu  = err_eu_t[-1];
    err_rk4 = err_rk4_t[-1];
    err_ty  = err_ty_t[-1];
    err_ty_m= err_tym_t[-1];

    names = ["Euler", "RK2 (Heun)", "RK4", "Taylor-10"];
    t_lte, L = lte_over_time(t_span=(0.0, 10.0), h=0.1, omega=1.0, order=10);

    # --- local error vs time
    plt.figure(figsize=(8,5))
    for k in range(L.shape[0]):
        plt.loglog(t_lte, L[k], marker='o', label=f"{names[k]} (h=0.1)")
    plt.xlabel("t (step start)")
    plt.ylabel(r"one-step LTE  $\|Y_{\rm num}(t\!\to\!t{+}h)-Y(t{+}h)\|_2$")
    plt.title("Local truncation error vs time (sine–cosine system)")
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout(); plt.show()

    # --- Global error vs time
    plt.figure(figsize=(8,5));
    plt.loglog(t_eu,  err_eu_t,  'o-', ms=3, label=f'Euler (h={h_euler})');
    plt.loglog(t_rk4, err_rk4_t, '^-', ms=4, label=f'RK4 (h={h_rk4})');
    plt.loglog(t_ty,  err_ty_t,  'd-', ms=4, label=f'Taylor-10 (h={h_taylor})');
    plt.loglog(t_ty_m,err_tym_t, 's-', ms=4, label=f'Matrix Taylor-10 (h={h_taylor})');
    plt.xlabel('t'); plt.ylabel(r'global error  $\|[y,z]_{\rm num}-[y,z]_{\rm exact}\|_2$');
    plt.title('Global error vs time'); plt.grid(True, which='both'); plt.legend();
    plt.tight_layout(); plt.show();


if __name__ == "__main__":
    main();

# End of file
