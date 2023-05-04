import numpy as np
import matplotlib.pyplot as plt

A,B,err = np.loadtxt('pyreg/minima.dat',delimiter=' ', unpack=True)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(A,B,err,'.-')
plt.show()