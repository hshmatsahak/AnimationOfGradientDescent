import numpy as np 
import matplotlib.pyplot as plt 
from celluloid import Camera
from matplotlib import animation

from NeuralNet_3 import M1, M2, Z_100
from NeuralNet_4 import weights_2_5_250, weights_2_5_251, costs

fig = plt.figure(figsize=(10,10)) # create figure
ax = fig.add_subplot(111,projection='3d')
line_style=["dashed", "dashdot", "dotted"] #linestyle
fontsize_=27 # set axis label fontsize
labelsize_=17 # set tick label fontsize
ax.view_init(elev=30, azim=-10)
ax.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=17)
ax.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=5)
ax.set_zlabel("costs", fontsize=fontsize_, labelpad=-35)
ax.tick_params(axis='x', pad=12, which='major', labelsize=labelsize_)
ax.tick_params(axis='y', pad=0, which='major', labelsize=labelsize_)
ax.tick_params(axis='z', pad=8, which='major', labelsize=labelsize_)
ax.set_zlim(4.75, 4.802) # set range for z-values in the plot

# Define which epochs to plot
p1=list(np.arange(0,200,20))
p2=list(np.arange(200,9000,100))
points_=p1+p2

camera=Camera(fig)
for i in points_:
    # Plot the three trajectories of gradient descent
    # ...each starting from its respective starting point
    # ...and each with a unique linestyle
    for j in range(3):
        ax.plot(weights_2_5_250[j][0:i],weights_2_5_251[j][0:i],costs[j][0:i],
                linestyle=line_style[j],linewidth=2,
                color="black", label=str(i))
        ax.scatter(weights_2_5_250[j][i],weights_2_5_251[j][i],costs[j][i],
                   marker='o', s=15**2,
               color="black", alpha=1.0)
    # Surface plot (= loss landscape)
    ax.plot_surface(M1, M2, Z_100, cmap='terrain',
                    antialiased=True, cstride=1, rstride=1, alpha=0.75)
    ax.legend([f'epochs: {i}'], loc=(0.25, 0.8), fontsize=17) # set position of legend
    plt.tight_layout()
    camera.snap() # take snapshot after each iteration

animation = camera.animate(interval=5, repeat=False, repeat_delay=0)
animation.save('gd_1.gif', writer='imagemagick', dpi=100)
    