import numpy as np
from numpy import linspace, random, array, zeros, pi
from PS_check_jit_param import ps_param, ps_reg
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.interpolate import griddata
from tqdm import tqdm
import time

def find_lambda(
                    trj_ellipse: np.ndarray,
                    kov_ellipse: np.ndarray,
                    bounds: tuple,
                    eps: float,
                    max_iter: int
               ) -> float:
    iter_indx = 0
    lower, upper = bounds
    while upper > lower and (iter_indx < max_iter if max_iter else True) and abs(upper - lower) > eps:
        lmbda = (upper + lower) / 2
        PS_check = ps_reg(lmbda * kov_ellipse[0], lmbda * kov_ellipse[1], trj_ellipse)

        #print(f"Iteration {iter_indx}: λ={lmbda}, lower={lower}, upper={upper}, PS_check={PS_check[0]}")

        if bool(PS_check[0]):
            lower = lmbda + eps
        else:
            upper = lmbda - eps
        iter_indx += 1
    
    final_lambda = (upper + lower) / 2
    #print(f"Final λ: {final_lambda}")
    return final_lambda

# print(find_lambda(np.array([2.101299999999999557e-01,1.400800000000000101e-01,0.000000000000000000e+00,0.00800000000000101e-01]),
#                   np.array([0.13, 0.08]),
#                   np.array([1e-12, 8.]),
#                   1e-6,
#                   10000))

# transition pt parameters
bounds = np.array([1e-12, 8.])
eps = 1e-6
max_iter = 1000

# standard ellipse parameters
As = 0.13
Bs = 0.08
kov = np.array([As, Bs])

# ellipse parameters
Ag = np.arange(As*1.001, As*2, 0.02)
Bg = np.arange(Bs*1.001, Bs*2, 0.02)
Psig = np.arange(0., pi, pi/180)
Kg = np.arange(-Bs*2, Bs*2, 0.001)
total_num = len(Ag) * len(Bg) * len(Psig) * len(Kg)

# Create sweep of ellipse parameters
ellipse_parameters_g = zeros((total_num, 4))
index = 0
for A in Ag:
    for B in Bg:
        for Psi in Psig:
            for K in Kg:
                ellipse_parameters_g[index, :] = [A, B, Psi, K]
                index += 1
ellipse_parameters_g = array(ellipse_parameters_g)

# Obtain roots of parameters and compute transition point
root_sol = zeros((total_num, 13))
for i, ellipse_param in enumerate(tqdm(ellipse_parameters_g, desc="Solving...")):
    root_sol[i, :12]    = ps_param(As, Bs, ellipse_param)
    root_sol[i, -1]     = find_lambda(ellipse_param, kov, bounds, eps, max_iter)

# Get second largest root
second_largest_roots = zeros((total_num, 5))
for i, root in enumerate(root_sol):
    values = root[[0, 4, 8]]
    sorted_indices = np.argsort(values)
    
    # Get the second largest value's original index
    original_index = [0, 4, 8][sorted_indices[-2]]
    second_largest = root[original_index]
    
    second_largest_roots[i, 0] = second_largest
    second_largest_roots[i, 1:4] = root[original_index:original_index+3]  # Corrected indexing
    second_largest_roots[i, -1] = root[-1]  # Last element remains the same


# Filter the array
filtered_indices = (second_largest_roots[:, 0] < 1) & (second_largest_roots[:, -1] < 1)
filtered_roots = root_sol[filtered_indices]
filtered_ellipse_parameters = ellipse_parameters_g[filtered_indices]
output_array = np.concatenate((filtered_roots, filtered_ellipse_parameters), axis=1)
np.savetxt('second_largest_roots.txt', output_array, delimiter=',')

# Save output to file
filtered_indices2           = second_largest_roots[:, 0] < 1.1
filtered_2ndroots           = second_largest_roots[filtered_indices2]
filtered_2ndroots_ellipse   = ellipse_parameters_g[filtered_indices2]
output_array2               = np.concatenate((filtered_2ndroots, filtered_2ndroots_ellipse), axis=1)
np.savetxt('output.txt', output_array2, delimiter=',')

# Filter to be below 1.1
filtered_indices2 = second_largest_roots[:, 0] < 1.1
filtered_2ndlarge_roots = second_largest_roots[filtered_indices2]

# Assuming second_largest_roots is a 2D numpy array
a = second_largest_roots[:, 0]
b = second_largest_roots[:, 1]
c = second_largest_roots[:, 2]
d = second_largest_roots[:, 3]
y = second_largest_roots[:, 4]

# Create a custom colormap
colors = [(1, 0.5, 0.5), (0, 0, 0), (1, 1, 0)]  # Light Red, Black, Yellow
n_bins = 100  # Discretize the colormap
cmap_name = 'light_red_black_yellow'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
norm = TwoSlopeNorm(vmin=y.min(), vcenter=1, vmax=y.max())
phi = np.linspace(0, 2*np.pi, 1000)

scatter = ax.scatter(d, a, c=y, cmap=cm, s=5, zorder=2, norm=norm)
ax.plot(phi, np.ones(1000), 'r-', lw=2, zorder=3)
ax.set_rlabel_position(180)  # Move radial labels away from plotted line
# ax.set_rmax(2)
ax.set_ylim((0., 1.5))
ax.set_rticks([0., 0.5, 1., 2.])  # Set radial ticks
ax.set_yticklabels(['0', '0.5', '1', '2'], color='red')  # Set tick labels and their color
for label in ax.get_yticklabels():
    label.set_bbox(dict(facecolor='white', edgecolor='none', pad=1))
    label.set_zorder(4)
ax.set_aspect('equal')
cbar = fig.colorbar(scatter, ax=ax, label=r'$\beta$', norm=norm)


plt.tight_layout()
plt.show()
