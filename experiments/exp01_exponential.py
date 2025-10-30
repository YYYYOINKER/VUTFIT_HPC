#
#   Experiment 01: Exponential growth/decay
#   Author: Pavol Mihalik, VUT FIT
#
#   Compare: Euler, RK2, RK4, specialized Taylor vs exact solution
#

import numpy as np;
import matplotlib.pyplot as plt;

# Import system
from src.systems import exponential;
from src.systems.exponential import local_error_all_methods, global_error_all_methods;

# Import general solvers
from src.integrators.euler import euler;
from src.integrators.rk2 import rk2;
from src.integrators.rk4 import rk4;

def main():

    # Problem setup
    lam: float = 2.6;
    y0: float = 1.0;
    t_span: tuple[float, float] = (0.0, 5.0);

    # Setup grid and exact solution
    t_dense: np.ndarray = np.linspace(t_span[0], t_span[1], 1000);
    y_exact_dense: np.ndarray = exponential.exact_solution(t_dense, y0, lam); 
    exact_final: float = exponential.exact_solution(t_span[1], y0, lam);

    # Individual step sizes for each solver
    h_euler: float = 0.2;
    h_rk2: float = 0.5;
    h_rk4: float = 0.7;
    h_taylor: float = 1.0;

    # Euler
    t_eu, y_eu = euler(lambda t, y: exponential.f(t, y, lam), t_span, y0, h_euler);
    err_eu = abs(y_eu[-1] - exact_final);

    # RK2
    t_rk2, y_rk2 = rk2(lambda t, y: exponential.f(t, y, lam), t_span, y0, h_rk2);
    err_rk2 = abs(y_rk2[-1] - exact_final);

    # RK4
    t_rk4, y_rk4 = rk4(lambda t, y: exponential.f(t, y, lam), t_span, y0, h_rk4);
    err_rk4 = abs(y_rk4[-1] - exact_final);

    # Taylor
    t_ty, y_ty = exponential.taylor_recursive_diff(t_span, y0, h_taylor, lam, order=10);
    err_ty = abs(y_ty[-1] - exact_final);

    h = 0.1;
    # local errors
    t_starts, ltes = local_error_all_methods(
        exponential.f, t_span=t_span, y0=y0, lam=lam, h=h
    );

    t_gerr, gerr = global_error_all_methods(exponential.f, t_span=t_span, 
                                            y0=y0, lam=lam, h=h
    );
    
    # Print results
    print(f"\nFinal time t = {t_span[1]}, exact = {exact_final:.6e}\n");
    print(f"{'Method':<15} | {'h':>8} | {'Final Error':>15}");
    print("-" * 45);
    print(f"{'Euler':<15} | {h_euler:8.3f} | {err_eu:15.6e}");
    print(f"{'RK2':<15} | {h_rk2:8.3f} | {err_rk2:15.6e}");
    print(f"{'RK4':<15} | {h_rk4:8.3f} | {err_rk4:15.6e}");
    print(f"{'Taylor n=10':<15} | {h_taylor:8.3f} | {err_ty:15.6e}");

    # Plot
    plt.figure(figsize=(8, 5));
    plt.plot(t_dense, y_exact_dense, "k--", label="Exact");
    plt.plot(t_eu, y_eu, "o-", label=f"Euler (h={h_euler})");
    plt.plot(t_rk2, y_rk2, "s-", label=f"RK2 (h={h_rk2})");
    plt.plot(t_rk4, y_rk4, "^-", label=f"RK4 (h={h_rk4})");
    plt.plot(t_ty, y_ty, "d-", label=f"Taylor n=10 (h={h_taylor})");
    plt.xlabel("t");
    plt.ylabel("y(t)");
    plt.grid();
    plt.legend();
    plt.title("Experiment 01: Exponential ODE y' = λy");
    

    # Save input data and output data to a text file
    with open('results/exp01/data.txt', 'a') as file:

        file.write(f"Problem: y' = λ y ; y(0) = 1, λ = {lam}, \ntime span: <{t_span[0]};{t_span[1]}>\n")
        file.write(f"{'Method':<15} | {'h':>8} | {'Final Error':>15}\n");
        file.write("-" * 45 + "\n");
        file.write(f"{'Euler':<15} | {h_euler:8.3f} | {err_eu:15.6e}\n");
        file.write(f"{'RK2':<15} | {h_rk2:8.3f} | {err_rk2:15.6e}\n");
        file.write(f"{'RK4':<15} | {h_rk4:8.3f} | {err_rk4:15.6e}\n");
        file.write(f"{'Taylor n=10':<15} | {h_taylor:8.3f} | {err_ty:15.6e}\n\n");

    plt.savefig('results/exp01/experiment01.png');

    plt.show();

    # --- plot LTE(t_i, h) vs t_i ---
    plt.figure(figsize=(8,5));
    plt.semilogy(t_starts, ltes[0], 'o-', label=f"Euler   (h={h})");
    plt.semilogy(t_starts, ltes[1], 's-', label=f"RK2     (h={h})");
    plt.semilogy(t_starts, ltes[2], '^-', label=f"RK4     (h={h})");
    plt.semilogy(t_starts, ltes[3], 'p-', label=f"Taylor(ord=10)    (h={h})");
    plt.xlabel("t (step start)");
    plt.ylabel("one-step LTE = |y_num(t→t+h) - y_exact(t+h)|");
    plt.title("Local truncation error along the interval");
    plt.grid(True, which="both"); plt.legend(); plt.tight_layout();
    plt.show();

    # --- plot GLOBALERR(t_i, h) vs t_i ---
    method_names = ["Euler", "RK2 (Heun)", "RK4", "Taylor-10"];
    plt.figure(figsize=(8,5));
    for m in range(gerr.shape[0]):
        plt.semilogy(t_gerr, gerr[m], marker=".", label=f"{method_names[m]} (h={h})");
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