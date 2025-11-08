import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data
exp = np.array([0.7636, 0.7812, 0.8032, 0.6228, 0.5598, 0.261])
hb = np.array([0.7823, 0.7938, 0.8027, 0.6115, 0.4386, 0.2691])
the = np.array([0.9491, 0.9084, 0.9326, 0.8430, 0.9336, 0.9092])

exp1 = exp
hybrid = hb
sim = the

# Error calculation
error1 = ((exp1 - hybrid) / exp1) * 100
error2 = ((exp1 - sim) / exp1) * 100

# Fit normal distributions
mu1, std1 = norm.fit(error1)
mu2, std2 = norm.fit(error2)

# Generate values for x-axis
x = np.linspace(-330, 330, 1000)

# Probability density functions
y1 = norm.pdf(x, mu1, std1)
y2 = norm.pdf(x, mu2, std2)

# Plotting
plt.plot(x, y1, 'b-', linewidth=2, label='Hybrid fuzzy model')
plt.plot(x, y2, 'r-', linewidth=2, label='Theoretical model')
plt.xlabel('Prediction error (%)', fontsize=23, fontname='Times New Roman')
plt.ylabel('Probability density', fontsize=23, fontname='Times New Roman')
plt.legend(frameon=False)
plt.grid(True)
plt.show()
