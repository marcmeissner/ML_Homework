import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

data = np.genfromtxt('01_homework_dataset.csv', delimiter=',', skip_header=1,
                     skip_footer=0, names=['x1', 'x2', 'x3', 'z'])
helper = data['x1']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x1'], data['x2'], data['x3'], c=data['z'])
ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('X3 Label')
ax.legend()
plt.show()