import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**2

def slope(x, dx):
    return (dx*x)

#Array of numbers between 0 and 5 with an increment of 0.001
x = np.array(np.arange(0,5,0.001))
y = f(x)

plt.plot(x, y)
plt.show()