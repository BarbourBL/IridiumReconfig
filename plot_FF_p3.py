import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

dir = "Files/"

# Extract Ellipse Param 1
sat1_ellip = []
with open(dir+"Sat1_EllipseParam.txt", "r") as SAT1_PARAM_FILE:
    lines = SAT1_PARAM_FILE.readlines()
    sat1_ellip.append([float(num) for num in lines[1].split()])
sat1_ellip = np.array(sat1_ellip).flatten()

# Extract Ellipse Param 2
sat2_ellip = []
with open(dir+"Sat2_EllipseParam.txt", "r") as SAT2_PARAM_FILE:
    lines = SAT2_PARAM_FILE.readlines()
    sat2_ellip.append([float(num) for num in lines[1].split()])
sat2_ellip = np.array(sat2_ellip).flatten()

# Extract Ellipse Param 3
sat3_ellip = []
with open(dir+"Sat3_EllipseParam.txt", "r") as SAT3_PARAM_FILE:
    lines = SAT3_PARAM_FILE.readlines()
    sat3_ellip.append([float(num) for num in lines[1].split()])
sat3_ellip = np.array(sat3_ellip).flatten()

# Extract RIC history 1
sat1_RIC_hist = []
with open(dir+"Sat1_RICPosition_data.txt", "r") as SAT1_RIC_FILE:
    lines = SAT1_RIC_FILE.readlines()
    for i in range(1, len(lines)-1):
        content = lines[i][1:-3]
        sat1_RIC_hist.append([float(num) for num in content.split()])
sat1_RIC_hist = np.array(sat1_RIC_hist)

# Extract RIC history 2
sat2_RIC_hist = []
with open(dir+"Sat2_RICPosition_data.txt", "r") as SAT2_RIC_FILE:
    lines = SAT2_RIC_FILE.readlines()
    for i in range(1, len(lines)-1):
        content = lines[i][1:-3]
        sat2_RIC_hist.append([float(num) for num in content.split()])
sat2_RIC_hist = np.array(sat2_RIC_hist)

# Extract RIC history 3
sat3_RIC_hist = []
with open(dir+"Sat3_RICPosition_data.txt", "r") as SAT3_RIC_FILE:
    lines = SAT3_RIC_FILE.readlines()
    for i in range(1, len(lines)-1):
        content = lines[i][1:-3]
        sat3_RIC_hist.append([float(num) for num in content.split()])
sat3_RIC_hist = np.array(sat3_RIC_hist)

# Extract ROE PS Solution 1
sat1_roe_sol = []
with open(dir+"Sat1_Sol_SE.txt", "r") as SAT1_SOL_FILE:
    lines = SAT1_SOL_FILE.readlines()
    sat1_roe_sol.append([float(num) for num in lines[1].split()])
sat1_roe_sol = np.array(sat1_roe_sol).flatten()
sat1_root_mag = np.array([np.linalg.norm(sat1_roe_sol[4:6]), np.linalg.norm(sat1_roe_sol[6:8]), np.linalg.norm(sat1_roe_sol[8:])])
sat1_root_phase = np.array([np.arctan2(sat1_roe_sol[5], sat1_roe_sol[4]), 
                            np.arctan2(sat1_roe_sol[7], sat1_roe_sol[6]), 
                            np.arctan2(sat1_roe_sol[9], sat1_roe_sol[8])])
sat1_indices = np.argsort(sat1_root_mag)[::-1]
sat1_2nd_index = sat1_indices[1]
sat1_pen_root_idx = 4 + 2 * sat1_2nd_index

# Extract ROE PS Solution 2
sat2_roe_sol = []
with open(dir+"Sat2_Sol_SE.txt", "r") as SAT2_SOL_FILE:
    lines = SAT2_SOL_FILE.readlines()
    sat2_roe_sol.append([float(num) for num in lines[1].split()])
sat2_roe_sol = np.array(sat2_roe_sol).flatten()
sat2_root_mag = np.array([np.linalg.norm(sat2_roe_sol[4:6]), np.linalg.norm(sat2_roe_sol[6:8]), np.linalg.norm(sat2_roe_sol[8:])])
sat2_root_phase = np.array([np.arctan2(sat2_roe_sol[5], sat2_roe_sol[4]), 
                            np.arctan2(sat2_roe_sol[7], sat2_roe_sol[6]), 
                            np.arctan2(sat2_roe_sol[9], sat2_roe_sol[8])])
sat2_indices = np.argsort(sat2_root_mag)[::-1]
sat2_2nd_index = sat2_indices[1]
sat2_pen_root_idx = 4 + 2 * sat2_2nd_index

# Extract ROE PS Solution 3
sat3_roe_sol = []
with open(dir+"Sat3_Sol_SE.txt", "r") as SAT3_SOL_FILE:
    lines = SAT3_SOL_FILE.readlines()
    sat3_roe_sol.append([float(num) for num in lines[1].split()])
sat3_roe_sol = np.array(sat3_roe_sol).flatten()
sat3_root_mag = np.array([np.linalg.norm(sat3_roe_sol[4:6]), np.linalg.norm(sat3_roe_sol[6:8]), np.linalg.norm(sat3_roe_sol[8:])])
sat3_root_phase = np.array([np.arctan2(sat3_roe_sol[5], sat3_roe_sol[4]), 
                            np.arctan2(sat3_roe_sol[7], sat3_roe_sol[6]), 
                            np.arctan2(sat3_roe_sol[9], sat3_roe_sol[8])])
sat3_indices = np.argsort(sat3_root_mag)[::-1]
sat3_2nd_index = sat3_indices[1]
sat3_pen_root_idx = 4 + 2 * sat3_2nd_index


# SMA = 7161.973616275
# DA = -15.797367331 / SMA
# DEX = 11.135375963 / SMA
# DEY = 4.114418905 / SMA
# DIX = 0.000000000 / SMA
# DIY = -0.000145734 / SMA
# PHI = np.arctan2(DEY, DEX)
# THETA = np.arctan2(DIY, DIX)
# DE = np.sqrt(DEX**2 + DEY**2)
# DI = np.sqrt(DIX**2 + DIY**2)
# PSIG0 = 1/2 * np.arctan2(2 * DE * DI * np.sin(PHI - THETA), DE**2 - DI**2)
# PSIG0 = PSIG0 + np.sign(PSIG0)*np.pi/2
# print(PSIG0)

# DA = -15.797874222 / SMA
# DEX = 11.629316323 / SMA
# DEY = 2.740439323 / SMA
# DIX = -0.024581622 / SMA
# DIY = -4.672998431 / SMA
# PHI = np.arctan2(DEY, DEX)
# THETA = np.arctan2(DIY, DIX)
# DE = np.sqrt(DEX**2 + DEY**2)
# DI = np.sqrt(DIX**2 + DIY**2)
# PSIG1 = 1/2 * np.arctan2(2 * DE * DI * np.sin(PHI - THETA), DE**2 - DI**2)
# PSIG1 = PSIG1 + np.sign(PSIG1)*np.pi/2
# print(PSIG1)

# DA = -15.798579847 / SMA
# DEX = 11.622945774 / SMA
# DEY = 1.379658954 / SMA
# DIX = -0.002907808 / SMA
# DIY = 0.740366581 / SMA
# PHI = np.arctan2(DEY, DEX)
# THETA = np.arctan2(DIY, DIX)
# DE = np.sqrt(DEX**2 + DEY**2)
# DI = np.sqrt(DIX**2 + DIY**2)
# PSIG2 = 1/2 * np.arctan2(2 * DE * DI * np.sin(PHI - THETA), DE**2 - DI**2)
# PSIG2 = PSIG2 + np.sign(PSIG2)*np.pi/2
# print(PSIG2)

SMA = 7161.973616275
DA = -4.478328499 / SMA
DEX = 6.571676988 / SMA
DEY = 2.430399053 / SMA
DIX = 6.574635527 / SMA
DIY = 2.434372884 / SMA
PHI = np.arctan2(DEY, DEX)
THETA = np.arctan2(DIY, DIX)
DE = np.sqrt(DEX**2 + DEY**2)
DI = np.sqrt(DIX**2 + DIY**2)
PSIG0 = 1/2 * np.arctan2(2 * DE * DI * np.sin(PHI - THETA), DE**2 - DI**2)
PSIG0 = PSIG0 + np.sign(PSIG0)*np.pi/2
print(PSIG0)

DA = -4.478835390 / SMA
DEX = 7.076327234 / SMA
DEY = -0.364370709 / SMA
DIX = 6.550054370 / SMA
DIY = 2.873299736 / SMA
PHI = np.arctan2(DEY, DEX)
THETA = np.arctan2(DIY, DIX)
DE = np.sqrt(DEX**2 + DEY**2)
DI = np.sqrt(DIX**2 + DIY**2)
PSIG1 = 1/2 * np.arctan2(2 * DE * DI * np.sin(PHI - THETA), DE**2 - DI**2)
PSIG1 = PSIG1 + np.sign(PSIG1)*np.pi/2
print(PSIG1)

DA = -4.479541015 / SMA
DEX = 6.178472064 / SMA
DEY = -2.945838672 / SMA
DIX = 6.571728832 / SMA
DIY = 13.436736630 / SMA
PHI = np.arctan2(DEY, DEX)
THETA = np.arctan2(DIY, DIX)
DE = np.sqrt(DEX**2 + DEY**2)
DI = np.sqrt(DIX**2 + DIY**2)
PSIG2 = 1/2 * np.arctan2(2 * DE * DI * np.sin(PHI - THETA), DE**2 - DI**2)
PSIG2 = PSIG2 + np.sign(PSIG2)*np.pi/2
print(PSIG2)


fig = plt.figure(figsize=(11, 4))

# Create top subplots with 3D projection
u = np.linspace(0, np.pi, 50)      # polar angle
v = np.linspace(0, 2 * np.pi, 50)  # azimuthal angle
u, v = np.meshgrid(u, v)
x = np.sin(u) * np.cos(v)
y = np.sin(u) * np.sin(v)
z = np.cos(u)

# ax_top_left = fig.add_subplot(1, 3, 1, projection='3d')
# ax_top_left.set_title(r"Satellite 1")
# ax_top_left.view_init(elev=11, azim=34, roll=0)
# ax_top_left.set_xlabel(r'In-Track (km)')
# ax_top_left.set_ylabel(r'Cross-Track (km)')
# ax_top_left.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.1, edgecolor='k')
# ax_top_left.set_box_aspect((1,1,1))

# ax_top_mid = fig.add_subplot(1, 3, 2, projection='3d')
# ax_top_mid.set_title(r"Satellite 2")
# ax_top_mid.view_init(elev=11, azim=34, roll=0)
# ax_top_mid.set_xlabel(r'In-Track (km)')
# ax_top_mid.set_ylabel(r'Cross-Track (km)')
# ax_top_mid.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.1, edgecolor='k')
# ax_top_mid.set_box_aspect((1,1,1))

# ax_top_right = fig.add_subplot(1, 3, 3, projection='3d')
# ax_top_right.set_title(r"Satellite 3")
# ax_top_right.view_init(elev=11, azim=34, roll=0)
# ax_top_right.set_xlabel(r'In-Track (km)')
# ax_top_right.set_ylabel(r'Cross-Track (km)')
# ax_top_right.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.1, edgecolor='k')
# ax_top_right.set_box_aspect((1,1,1))

# Create bottom subplots as standard 2D axes
ax_bottom_left = fig.add_subplot(1, 3, 1)
ax_bottom_mid = fig.add_subplot(1, 3, 2)
ax_bottom_right = fig.add_subplot(1, 3, 3)

# 3D:
# ax_top_left.plot(sat1_RIC_hist[:, 1], sat1_RIC_hist[:, 2], sat1_RIC_hist[:, 0], 'k-', alpha=0.5)
# ax_top_left.scatter(0, 0, 0, c='k', s=80, marker='+')
# ax_top_mid.plot(sat2_RIC_hist[:, 1], sat2_RIC_hist[:, 2], sat2_RIC_hist[:, 0], 'k-', alpha=0.5)
# ax_top_mid.scatter(0, 0, 0, c='k', s=80, marker='+')
# ax_top_right.plot(sat3_RIC_hist[:, 1], sat3_RIC_hist[:, 2], sat3_RIC_hist[:, 0], 'k-', alpha=0.5)
# ax_top_right.scatter(0, 0, 0, c='k', s=80, marker='+')


# 2D Ellipse:
ax_bottom_left.add_patch(Ellipse((0, sat1_ellip[3]*1e3), sat1_ellip[0]*2*1e3, sat1_ellip[1]*2*1e3, angle=np.rad2deg(PSIG0), edgecolor='b', facecolor='none', lw=2))
ax_bottom_mid.add_patch(Ellipse((0, sat2_ellip[3]*1e3), sat2_ellip[0]*2*1e3, sat2_ellip[1]*2*1e3, angle=np.rad2deg(PSIG1), edgecolor='b', facecolor='none', lw=2))
ax_bottom_right.add_patch(Ellipse((0, sat3_ellip[3]*1e3), sat3_ellip[0]*2*1e3, sat3_ellip[1]*2*1e3, angle=np.rad2deg(PSIG2), edgecolor='b', facecolor='none', lw=2))

ax_bottom_left.add_patch(Ellipse((0, 0), 2*1e3, 2*1e3, angle=0, edgecolor='k', facecolor='none', lw=1))
ax_bottom_mid.add_patch(Ellipse((0, 0), 2*1e3, 2*1e3, angle=0, edgecolor='k', facecolor='none', lw=1))
ax_bottom_right.add_patch(Ellipse((0, 0), 2*1e3, 2*1e3, angle=0, edgecolor='k', facecolor='none', lw=1))

# 2D:
ax_bottom_left.set_title(r"Satellite 1")
ax_bottom_left.plot(sat1_RIC_hist[:, 2]*1e3, sat1_RIC_hist[:, 0]*1e3, 'k-', alpha=0.5, zorder=10)
ax_bottom_left.scatter(0, 0, c='k', s=20, marker='+')
ax_bottom_left.set_ylabel(r'Radial (m)')
ax_bottom_left.set_xlabel(r'Cross-Track (m)')

ax_bottom_mid.set_title(r"Satellite 2")
ax_bottom_mid.plot(sat2_RIC_hist[:, 2]*1e3, sat2_RIC_hist[:, 0]*1e3, 'k-', alpha=0.5)
ax_bottom_mid.scatter(0, 0, c='k', s=20, marker='+')
ax_bottom_mid.set_xlabel(r'Cross-Track (m)')

ax_bottom_right.set_title(r"Satellite 3")
ax_bottom_right.plot(sat3_RIC_hist[:, 2]*1e3, sat3_RIC_hist[:, 0]*1e3, 'k-', alpha=0.5)
ax_bottom_right.scatter(0, 0, c='k', s=20, marker='+')
ax_bottom_right.set_xlabel(r'Cross-Track (m)')

# ax_bottom_left.set_aspect('equal')
# ax_bottom_mid.set_aspect('equal')
# ax_bottom_right.set_aspect('equal')
# ax_bottom_left.set_xlim((min(sat2_RIC_hist[:, 2])*1e3*1.1, max(sat2_RIC_hist[:, 2])*1e3*1.1))
# ax_bottom_left.set_ylim((min(sat1_RIC_hist[:, 0])*1e3, max(sat1_RIC_hist[:, 0])*1e3))
# ax_bottom_mid.set_xlim((min(sat2_RIC_hist[:, 2])*1e3*1.1, max(sat2_RIC_hist[:, 2])*1e3*1.1))
# ax_bottom_mid.set_ylim((min(sat1_RIC_hist[:, 0])*1e3, max(sat1_RIC_hist[:, 0])*1e3))
# ax_bottom_right.set_xlim((min(sat2_RIC_hist[:, 2])*1e3*1.1, max(sat2_RIC_hist[:, 2])*1e3*1.1))
# ax_bottom_right.set_ylim((min(sat1_RIC_hist[:, 0])*1e3, max(sat1_RIC_hist[:, 0])*1e3))

plt.tight_layout()



fig2, ax2 = plt.subplots(1,3, figsize=((11,5)), subplot_kw={'projection': 'polar'})
px = np.linspace(0, 2*np.pi, 100)
pm = np.ones((100))
#ax2[0].set_title(r"Satellite 1")
ax2[0].plot(px, pm, 'k-', lw=3, zorder=10)
ax2[0].scatter(sat1_root_phase[0], sat1_root_mag[0], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[0].scatter(sat1_root_phase[1], sat1_root_mag[1], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[0].scatter(sat1_root_phase[2], sat1_root_mag[2], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[0].scatter(sat1_root_phase[sat1_2nd_index], sat1_root_mag[sat1_2nd_index], marker='o', facecolor='cyan', edgecolor='k', s=70, zorder=10)
ax2[0].set_ylim((0,1.3))
ax2[0].fill_between(px, 0, 1, color='green', alpha=0.1)
ax2[0].fill_between(px, 1, 1.3, color='red', alpha=0.1)
ax2[0].set_rlabel_position(0)
ax2[0].set_rticks([0.5, 1.0, 1.3])

#ax2[1].set_title(r"Satellite 2")
ax2[1].plot(px, pm, 'k-', lw=3, zorder=10)
ax2[1].scatter(sat2_root_phase[0], sat2_root_mag[0], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[1].scatter(sat2_root_phase[1], sat2_root_mag[1], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[1].scatter(sat2_root_phase[2], sat2_root_mag[2], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[1].scatter(sat2_root_phase[sat2_2nd_index], sat2_root_mag[sat2_2nd_index], marker='o', facecolor='cyan', edgecolor='k', s=70, zorder=10)
ax2[1].set_ylim((0,1.3))
ax2[1].fill_between(px, 0, 1, color='green', alpha=0.1)
ax2[1].fill_between(px, 1, 1.3, color='red', alpha=0.1)
ax2[1].set_rlabel_position(0)
ax2[1].set_rticks([0.5, 1.0, 1.3])

#ax2[2].set_title(r"Satellite 3")
ax2[2].plot(px, pm, 'k-', lw=3, zorder=10)
ax2[2].scatter(sat3_root_phase[0], sat3_root_mag[0], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[2].scatter(sat3_root_phase[1], sat3_root_mag[1], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[2].scatter(sat3_root_phase[2], sat3_root_mag[2], marker='o', facecolor='gray', edgecolor='k', s=70, zorder=10)
ax2[2].scatter(sat3_root_phase[sat3_2nd_index], sat3_root_mag[sat3_2nd_index], marker='o', facecolor='cyan', edgecolor='k', s=70, zorder=10)
ax2[2].set_ylim((0,1.3))
ax2[2].fill_between(px, 0, 1, color='green', alpha=0.1)
ax2[2].fill_between(px, 1, 1.3, color='red', alpha=0.1)
ax2[2].set_rlabel_position(0)
ax2[2].set_rticks([0.5, 1.0, 1.3])

plt.tight_layout()
plt.show()
