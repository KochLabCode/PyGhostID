import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Vector field
# -------------------------------------------------
def theta_dot(theta1, theta2, K, n, pS):
    """
    Returns theta1_dot, theta2_dot on a grid
    """
    t1 = theta1 + pS
    t2 = theta2 + pS

    theta1_dot = (
        1 - np.cos(t1)
        + (1 + np.cos(t1)) * (n + K * (1 - np.cos(t2)))
    )

    theta2_dot = (
        1 - np.cos(t2)
        + (1 + np.cos(t2)) * (n + K * (1 - np.cos(t1)))
    )

    return theta1_dot, theta2_dot


# -------------------------------------------------
# Nullcline computation
# -------------------------------------------------
def compute_nullclines(K, n, pS, N=400):
    """
    Computes nullclines on [0, 2pi] x [0, 2pi]

    Returns:
        T1, T2 : meshgrid
        f1, f2 : theta1_dot, theta2_dot
    """
    theta = np.linspace(0, 2*np.pi, N)
    T1, T2 = np.meshgrid(theta, theta)

    f1, f2 = theta_dot(T1, T2, K, n, pS)

    return T1, T2, f1, f2


# -------------------------------------------------
# Plotting
# -------------------------------------------------
def plot_nullclines(K, n, pS, N=400):
    T1, T2, f1, f2 = compute_nullclines(K, n, pS, N)

    plt.figure(figsize=(6, 6))

    # theta1' = 0 nullcline
    plt.contour(
        T1, T2, f1,
        levels=[0],
        colors="C0",
        linewidths=2,
        label=r"$\dot\theta_1 = 0$"
    )

    # theta2' = 0 nullcline
    plt.contour(
        T1, T2, f2,
        levels=[0],
        colors="C1",
        linewidths=2,
        linestyles="--",
        label=r"$\dot\theta_2 = 0$"
    )

    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 2*np.pi)

    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(fr"Nullclines ($K={K},\, n={n},\, pS={pS}$)")

    plt.xticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])
    plt.yticks([0, np.pi, 2*np.pi], ["0", r"$\pi$", r"$2\pi$"])

    # manual legend handles (contour has no default legend)
    plt.plot([], [], "C0", label=r"$\dot\theta_1=0$")
    plt.plot([], [], "C1--", label=r"$\dot\theta_2=0$")
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    K = -1.4
    n = 0.5
    pS = 0

    plot_nullclines(K, n, pS)
