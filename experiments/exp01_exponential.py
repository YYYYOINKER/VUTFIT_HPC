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

# Import general solvers
from src.integrators.euler import euler;
from src.integrators.rk2 import rk2;
from src.integrators.rk4 import rk4;

def main():

    # Problem setup
    lam: float = 2.6;
    y0: float = 1.0;
    t_span: tuple[float, float] = (0.0, 3.0);

    # Setup grid and exact solution
    t_dense: np.ndarray = np.linspace(t_span[0], t_span[1], 1000);
    y_exact_dense: np.ndarray = exponential.exact_solution(t_dense, y0, lam); 
    exact_final: float = exponential.exact_solution(t_span[1], y0, lam);

    # Individual step sizes for each solver
    h_euler: float = 0.2;
    h_rk2: float = 0.5;
    h_rk4: float = 0.8;
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
    plt.show();

    # TODO 
    # Automatic export of result image to Experiments/Experiment01
    # Automatic export of input and output data to -||-
    #  

if __name__ == "__main__":
    main();

# End of file