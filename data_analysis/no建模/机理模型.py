import numpy as np
from scipy.special import erf

# Constants
Ns = 31
r1 = 42e-3
r2 = 146e-3
h = 20e-3
P = 101.325  # kPa
H = 709.1735  # Kpa*m3/mol

# Liquid flow rate
l = 20e-3 / 3600  # l/s

# Specific surface area of packing
a = 500  # m2/m3

# Volume of packing
v = np.pi * (r2**2 - r1**2) * (20e-3)

# Cross-sectional area
aa = np.pi * (1.6e-3)**2  # 16#

L = l / aa  # m/s

# Rotational speed distribution
rpm = 1200

# Superoxygen concentration
c_oo2 = 7e-9  # mol/l

# Rate constant k_no-ooh
k_nooo2 = 6.9e9

# Apparent rate constant
k_appi = k_nooo2 * c_oo2

r = np.sqrt(r1 * r2)
D_no = 2.21e-9

gn2 = 2 / 3600
y_in = 0.0005

# Model simulation output results
# Constants expression
a1 = (1 - y_in) / y_in
a2 = a * np.pi * h * (r2**2 - r1**2) / gn2

# Liquid film renewal time
t_new = (r2 - r1) / (0.02107 * (L**0.2279) * ((r * rpm)**0.5448) * 31)

# Liquid film mass transfer coefficient
kl1 = (np.sqrt(D_no * k_appi) / t_new) * (t_new * erf(np.sqrt(k_appi * t_new)) +
      np.sqrt((t_new / np.pi) / k_appi) * np.exp(-k_appi * t_new) +
      (0.5 / k_appi) * erf(np.sqrt(k_appi * t_new)))

ky_cal = 0.082 * 298.15 * 0.2700 * kl1
yno = 1 / (a1 * np.exp(a2 * ky_cal) + 1)

print(yno)
remove = (y_in - yno)/y_in
print(remove)