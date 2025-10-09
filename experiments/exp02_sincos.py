#
#   Experiment 01: Exponential growth/decay
#   Author: Pavol Mihalik, VUT FIT
#
#   Compare: Euler, RK2, RK4, specialized Taylor vs exact solution
#

import numpy as np;
import matplotlib.pyplot as plt;

# Import system
from src.systems import sin_cos;
from src.systems.sin_cos import euler;
from src.systems.sin_cos import rk4;
from src.systems.sin_cos import taylor_recursive_diff;
# TODO matarix calcualtions

def main():

    # Problem setup
    omega: float = 1.0;
    y0: float = 0.0;
    z0: float = 1.0;
    t_span: tuple[float, float] = (0.0, 20.0);

    # Step sizes
    h_euler: float = 0.01;
    h_rk4: float = 0.5;
    h_taylor: float = 0.8;

    # Euler
    t_eu, y_eu, z_eu = euler(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_euler, omega);

    # RK4
    t_rk4, y_rk4, z_rk4 = rk4(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_rk4, omega);

    # Taylor
    t_ty, y_ty, z_ty = taylor_recursive_diff(sin_cos.f_y, sin_cos.f_z, t_span, y0, z0, h_taylor, omega, order=10);

    # Exact solution at final time
    t_final = t_span[1];
    y_exact_final = z0 * np.sin(omega * t_final);
    z_exact_final = z0 * np.cos(omega * t_final);

    # Errors in y at final time
    err_eu = abs(y_eu[-1] - y_exact_final);
    err_rk4 = abs(y_rk4[-1] - y_exact_final);
    err_ty  = abs(y_ty[-1]  - y_exact_final);

    # Print results
    print(f"\nFinal time t = {t_final}, exact y = {y_exact_final:.6e}");
    print(f"{'Method':<15} | {'h':>8} | {'Final Error':>15}");
    print("-" * 45);
    print(f"{'Euler':<15} | {h_euler:8.3f} | {err_eu:15.6e}");
    print(f"{'RK4':<15} | {h_rk4:8.3f} | {err_rk4:15.6e}");
    print(f"{'Taylor n=10':<15} | {h_taylor:8.3f} | {err_ty:15.6e}");

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

    # Exact curve
    t_dense = np.linspace(*t_span, 500);
    y_exact = z0 * np.sin(omega * t_dense);
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

if __name__ == "__main__":
    main();

# End of file
