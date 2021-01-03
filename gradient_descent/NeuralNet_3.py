import numpy as np 
from NeuralNet_2 import costs
from NeuralNet_1 import X_train, y_train_oh
import matplotlib.pyplot as plt

# Set range of values for meshgrid
m1s = np.linspace(-15, 17, 40)
m2s = np.linspace(-15, 18, 40)
M1, M2= np.meshgrid(m1s, m2s) # create meshgrid

# Determine costs for each coordinate in meshgrid: 
zs_100 = np.array([costs(X_train[0:100], y_train_oh[0:100].T, np.array([[mp1]]), np.array([[mp2]]),135) for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_100 = zs_100.reshape(M1.shape) # z-values for N=100

zs_10000 = np.array([costs(X_train[0:10000], y_train_oh[0:10000].T, np.array([[mp1]]), np.array([[mp2]]),135) for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_10000 = zs_10000.reshape(M1.shape) # z-values for N=10,000

fontsize_=20 # set axis label fontsize
labelsize_=12 # set tick label size

# Plot loss landscapes
fig = plt.figure(figsize=(10,7.5)) # create figure
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')

# Customize subplots
ax0.view_init(elev=30, azim=-20)
ax0.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=9)
ax0.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-5)
ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
ax0.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
ax0.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
ax0.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
ax0.set_title('N:100', y=0.85, fontsize=15) # set title of subplot

ax1.view_init(elev=30, azim=-30)
ax1.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=9)
ax1.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-5)
ax1.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
ax1.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
ax1.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
ax1.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
ax1.set_title('N:10,000', y=0.85, fontsize=15)

# Surface plots of costs (=loss landscapes):
ax0.plot_surface(M1, M2, Z_100, cmap='terrain', #surface plot
                            antialiased=True, cstride=1, rstride=1, alpha=0.75)
ax1.plot_surface(M1, M2, Z_10000, cmap='terrain', #surface plot
                            )
plt.tight_layout()
plt.show()