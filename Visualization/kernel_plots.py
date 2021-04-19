import numpy as np
import pandas as pd


from numpy.linalg import norm

from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LinearLocator



def plot_3d_kernel_function():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_2d_kernel_function():


    r = np.linspace(-5, 5, num=2000)

    gaussian_kernel = lambda r, sigma: np.exp(-sigma * r ** 2)
    laplacian_kernel = lambda r, sigma: np.exp(-sigma * np.abs(r))

    kernel_function_dict ={
    "Gaussian kernel" : gaussian_kernel,
    "Laplacian Kernel" : laplacian_kernel
    }

    for key in kernel_function_dict.keys():
        for sigma in [10, 1, 0.5, 0.2]:
            kernel = kernel_function_dict[key]
            kernel(r, sigma)
            a = kernel(r, sigma)
            plt.plot(a, label="$\sigma={}$".format(sigma))
        plt.legend()
        plt.grid()
        plt.xlabel("$r = x - x'$")
        plt.title(key)
        #plt.ylabel('$t-SNE_{2}$')
        plt.show()
        plt.close()


    Ratio_quad_kernel = lambda r, c, beta: (r**2 + c)**(-beta)
    for beta in [1, 0.2]:
        for c in [10, 1, 0.2]:
            a = Ratio_quad_kernel(r, c, beta)
            plt.plot(a, label="$ Beta={beta}, c={c}$".format(beta=beta, c=c))
        plt.legend()
        plt.grid()
        plt.xlabel("$r = x - x'$")
        plt.title("Rational Quadratic Kernel")
        #plt.ylabel('$t-SNE_{2}$')
        plt.show()
        plt.close()




if __name__ == "__main__":
    plot_2d_kernel_function()

