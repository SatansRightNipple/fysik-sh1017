# MW 2023-03-20
# Python calculation of wave interference
# length unit: mm

import numpy as np
import matplotlib.pyplot as plt

# wave parameters
wavelength = 0.0005 # mm = 5000 Ångström, approx wavelength for green laser pointer
k = 2*np.pi/wavelength

# geometry parameters
d = 0.5 # distance between double slits
D = 2000 # distance to detector screen
W = 100 # width of detector screen
points = 1000 # number of pixels of detector screen

# single slit parameters
N = 10 # number of sources
w = 0.05 # width of the slit


############ Double ##############

# source positions for single slit
source_pos = np.linspace(-w/2,w/2, int(N/2))

# source positions for double slit
source1_pos = source_pos + d/2
source2_pos = source_pos - d/2

source_pos = np.concatenate((source1_pos, source2_pos), axis=None)

# compute time averaged intensity on a detector screen
screen = np.empty(points)
double_slit = np.empty(points)
intensity_double = np.empty(points)

for i in range(points):
    x = W*(i/(points-1)-0.5)
    screen[i] = x
    r = np.sqrt(D**2 + (x)**2)
    
    rVec = [None]*N
    for m in range(N):
        print(m)
        rVec[m] = np.sqrt(D**2 + (x-source_pos[m])**2)

    sum1 = np.float64(N/2)
    for l in range(N-1):
        for n in range(l+1,N):
            sum1+= np.cos(k*(rVec[l]-rVec[n]))

    double_slit[i] = sum1/r**2

# normalize the intensity patterns to have the same central peak amplitude
intensity_double = double_slit / np.max(double_slit)

############ Single ##############

# compute time averaged intensity on a detector screen
N=10
d = 0.05/(N-1) # separation between slits
# source positions
pos = np.linspace(-w/2, w/2, N)

# compute time averaged intensity on a detector screen
screen = np.empty(points)
intensity_single = np.empty(points)
for i in range(points):
    x = W*(i/(points-1)-0.5)
    screen[i] = x
    r = np.sqrt(D**2 + x**2)

    rVec = [None]*N
    for m in range(N):
        rVec[m] = np.sqrt(D**2 + (x-pos[m])**2)

    sum = np.float64(N/2)
    for l in range(N-1):
        for n in range(l+1,N):
            sum += np.cos(k*(rVec[l]-rVec[n]))

    intensity_single[i] = sum/r**2
intensity_single = intensity_single / max(intensity_single)

# plot the results
plt.plot(screen,intensity_single, label='Single Slit')
plt.plot(screen,intensity_double, label='Double Slit')

plt.xlabel('x')
plt.ylabel('intensity')
plt.legend()
plt.show()
