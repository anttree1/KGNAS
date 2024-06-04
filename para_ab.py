import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

x = np.linspace(0.1, 0.9, 9)
y = np.linspace(0.1, 0.9, 9)
X, Y = np.meshgrid(x, y)

Z = np.array([[0.13, 0.19, 0.24, 0.25, 0.16, 0.18, 0.20, 0.16, 0.06],
              [0.21, 0.01, 0.23, 0.11, 0.14, 0.09, 0.21, 0.0, 0.33],
              [0.38, 0.37, 0.43, 0.44, 0.32, 0.46, 0.31, 0.47, 0.56],
              [0.56, 0.63, 0.54, 0.31, 0.65, 0.74, 0.62, 0.54, 0.63],
              [0.61, 0.68, 0.79, 0.47, 0.67, 0.65, 0.72, 0.91, 0.09],
              [0.69, 0.57, 0.89, 0.62, 0.50, 0.88, 0.76, 0.72, 0.66],
              [0.72, 0.77, 0.74, 0.82, 0.85, 0.97, 1, 0.94, 0.75],
              [0.65, 0.64, 0.58, 0.7, 0.89, 0.42, 0.63, 0.76, 0.45],
              [0.63, 0.69, 0.54, 0.76, 0.75, 0.4, 0.53, 0.63, 0.21]])

xnew = np.linspace(0.1, 0.9, 100)
ynew = np.linspace(0.1, 0.9, 100)
Xnew, Ynew = np.meshgrid(xnew, ynew)
Znew = griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xnew, Ynew), method='cubic')
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xnew, Ynew, Znew, cmap=cm.viridis, edgecolor='none', antialiased=True)
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.04)
cbar.set_label('', rotation=270, labelpad=18)
ax.set_xlabel('\u03B2', fontsize=14)
ax.set_ylabel('\u03B1', fontsize=14)
ax.set_zlabel('Value', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True)
plt.savefig("high_res_surface_plot_improved.png", dpi=300)
plt.show()