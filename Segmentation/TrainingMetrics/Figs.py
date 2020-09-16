import numpy as np
import matplotlib.pyplot as plt

# 1D
plt.plot(np.loadtxt('1D_accuracy.txt'))
plt.plot(np.loadtxt('1D_val_accuracy.txt'))
plt.show()

# 2D
plt.plot(np.loadtxt('2D_accuracy.txt'))
plt.plot(np.loadtxt('2D_val_accuracy.txt'))
plt.show()
