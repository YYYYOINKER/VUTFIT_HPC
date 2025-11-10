#
#   Experiment 01: Exponential growth/decay
#   Author: Pavol Mihalik, VUT FIT
#
#   Compare: Euler, RK2, RK4, specialized Taylor vs exact solution
#

import numpy as np;
import matplotlib.pyplot as plt;
import time;

# Import system
from src.systems import exponential;
from src.systems.exponential import local_error_all_methods, global_error_all_methods;

# Import general solvers
from src.integrators.euler import euler;
from src.integrators.rk2 import rk2;
from src.integrators.rk4 import rk4;

from scipy.integrate import solve_ivp;


def main():

    # Problem setup
    lam: float = 1.1;
    y0: float = 3.2;
    t_span: tuple[float, float] = (0.0, 5.0);

    # Setup grid and exact solution
    t_dense: np.ndarray = np.linspace(t_span[0], t_span[1], 1000);
    y_exact_dense: np.ndarray = exponential.exact_solution(t_dense, y0, lam); 
    exact_final: float = exponential.exact_solution(t_span[1], y0, lam);

    # Individual step sizes for each solver
    h_euler: float = 0.01;
    h_rk2: float = 0.02;
    h_rk4: float = 0.04;
    h_taylor: float = 0.1;


    # Euler
    time_euler = time.perf_counter();
    t_eu, y_eu = euler(lambda t, y: exponential.f(t, y, lam), t_span, y0, h_euler);
    err_eu = abs(y_eu[-1] - exact_final);
    time_euler_final = time.perf_counter() - time_euler;

    # RK2
    time_rk2 = time.perf_counter();
    t_rk2, y_rk2 = rk2(lambda t, y: exponential.f(t, y, lam), t_span, y0, h_rk2);
    err_rk2 = abs(y_rk2[-1] - exact_final);
    time_rk2_final = time.perf_counter() - time_rk2;

    # RK4
    time_rk4 = time.perf_counter();
    t_rk4, y_rk4 = rk4(lambda t, y: exponential.f(t, y, lam), t_span, y0, h_rk4);
    err_rk4 = abs(y_rk4[-1] - exact_final);
    time_rk4_final = time.perf_counter() - time_rk4;

    # Taylor
    time_taylor = time.perf_counter();
    t_ty, y_ty = exponential.taylor_recursive_diff(t_span, y0, h_taylor, lam, order=10);
    err_ty = abs(y_ty[-1] - exact_final);
    time_taylor_final = time.perf_counter() - time_taylor; 

    # --- SciPy solve_ivp (adaptive RK45) -----------------------------------------
    def rhs(t, y):  # matches your f(t, y, lam)
        return exponential.f(t, y, lam)
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/h_rk4)+1)  # for plotting
    t_scipy_start = time.perf_counter()
    sol = solve_ivp(rhs, t_span, np.array([y0]), method="RK45",
                t_eval=t_eval, rtol=1e-9, atol=1e-12, max_step=h_rk4)
    t_scipy_final = time.perf_counter() - t_scipy_start

    # last value at final time (solve_ivp returns (n, m) shape)
    y_scipy = sol.y[0]
    err_scipy = abs(y_scipy[-1] - exact_final)

    h = 0.1;
    # local errors
    t_starts, ltes = local_error_all_methods(
        exponential.f, t_span=t_span, y0=y0, lam=lam, h=h
    );

    t_gerr, gerr = global_error_all_methods(exponential.f, t_span=t_span, 
                                            y0=y0, lam=lam, h=h
    );
    y_exact_grid = exponential.exact_solution(t_gerr, y0, lam);
    gerr_rel = gerr / np.maximum(1e-300, np.abs(y_exact_grid));  # avoid division by 0   

    def rel(err, exact): 
        return err/abs(exact) if exact != 0 else float('nan')
    
    rel_eu  = rel(err_eu, exact_final)
    rel_rk2 = rel(err_rk2, exact_final)
    rel_rk4 = rel(err_rk4, exact_final)
    rel_ty  = rel(err_ty,  exact_final)
    rel_sp  = rel(err_scipy, exact_final)
    
    # --- Print results ------------------------------------------------------------
    print(f"{'Method':<18} | {'h':>8} | {'Abs Error':>12} | {'Rel Error':>10} | {'Time [s]':>9}")
    print("-" * 70)
    print(f"{'Euler':<18} | {h_euler:8.3f} | {err_eu:12.3e} | {rel_eu:10.3e} | {time_euler_final:9.4f}")
    print(f"{'RK2':<18} | {h_rk2:8.3f} | {err_rk2:12.3e} | {rel_rk2:10.3e} | {time_rk2_final:9.4f}")
    print(f"{'RK4':<18} | {h_rk4:8.3f} | {err_rk4:12.3e} | {rel_rk4:10.3e} | {time_rk4_final:9.4f}")
    print(f"{'Taylor n=10':<18} | {h_taylor:8.3f} | {err_ty:12.3e} | {rel_ty:10.3e} | {time_taylor_final:9.4f}")
    print(f"{'solve_ivp (RK45)':<18} | {'adaptive':>8} | {err_scipy:12.3e} | {rel_sp:10.3e} | {t_scipy_final:9.4f}")
    print(f"  SciPy stats: nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")



    # Plot
    plt.figure(figsize=(8, 5));
    plt.plot(t_dense, y_exact_dense, "k--", label="Exact");
    plt.plot(t_eu, y_eu, "o-", label=f"Euler (h={h_euler})");
    plt.plot(t_rk2, y_rk2, "s-", label=f"RK2 (h={h_rk2})");
    plt.plot(t_rk4, y_rk4, "^-", label=f"RK4 (h={h_rk4})");
    plt.plot(t_ty, y_ty, "d-", label=f"Taylor n=10 (h={h_taylor})");
    plt.plot(sol.t, y_scipy, "-",  label="solve_ivp RK45 (adaptive)")
    plt.xlabel("t");
    plt.ylabel("y(t)");
    plt.grid();
    plt.legend();
    plt.title("Experiment 01: Exponential ODE y' = λy");
    

    # Save input data and output data to a text file
    with open('results/exp01/data.txt', 'a') as file:
        file.write(f"Problem: y' = λ y ; y(0) = 1, λ = {lam}, \ntime span: <{t_span[0]};{t_span[1]}>\n");
        file.write(f"{'Method':<18} | {'h':>8} | {'Final Error':>15} | {'Time [s]':>9}\n");
        file.write("-" * 60 + "\n");
        file.write(f"{'Euler':<18} | {h_euler:8.3f} | {err_eu:15.6e} | {time_euler_final:9.4f}\n");
        file.write(f"{'RK2':<18} | {h_rk2:8.3f} | {err_rk2:15.6e} | {time_rk2_final:9.4f}\n");
        file.write(f"{'RK4':<18} | {h_rk4:8.3f} | {err_rk4:15.6e} | {time_rk4_final:9.4f}\n");
        file.write(f"{'Taylor n=10':<18} | {h_taylor:8.3f} | {err_ty:15.6e} | {time_taylor_final:9.4f}\n");
        file.write(f"{'solve_ivp (RK45)':<18} | {'adaptive':>8} | {err_scipy:15.6e} | {t_scipy_final:9.4f}\n");
        file.write(f"  SciPy stats: nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}\n\n");


    plt.savefig('results/exp01/experiment01.png');

    plt.show();

    # --- plot LTE(t_i, h) vs t_i ---
    plt.figure(figsize=(8,5));
    plt.plot(t_starts, ltes[0], 'o-', label=f"Euler   (h={h})");
    plt.plot(t_starts, ltes[1], 's-', label=f"RK2     (h={h})");
    plt.plot(t_starts, ltes[2], '^-', label=f"RK4     (h={h})");
    plt.plot(t_starts, ltes[3], 'p-', label=f"Taylor(ord=10)    (h={h})");
    plt.xlabel("t (step start)");
    plt.ylabel("one-step LTE = |y_num(t→t+h) - y_exact(t+h)|");
    plt.title("Local truncation error along the interval");
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout();
    plt.show();

    # --- plot GLOBALERR(t_i, h) vs t_i ---
    method_names = ["Euler", "RK2 (Heun)", "RK4", "Taylor-10"];
    plt.figure(figsize=(8,5));
    for m in range(gerr.shape[0]):
        plt.plot(t_gerr, gerr_rel[m], marker=".", label=f"{method_names[m]} (h={h})");
    plt.xlabel("t");
    plt.ylabel(r"global error  $|y_h(t_n)-y(t_n)|$");
    plt.title("Global error vs. time");
    plt.grid(True, which="both");
    plt.legend();
    plt.tight_layout();
    plt.show();

if __name__ == "__main__":
    main();

# End of file