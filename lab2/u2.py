import numpy as np
import matplotlib.pyplot as plt
import statistics

#wavelength
wavelength = 0.0005 # mm = 5000 Ångström, approx wavelength för green laser pointer
k = 2*np.pi/wavelength

# geometry parameters
d = 0.05 # separation between slits
D = 200 # distance to detector screen
W = 200 # width of detector screen
points = 1000 # number of pixels of detector screen

# source positions
x1 = d/2
x2 = -d/2

# compute time averaged intensity on a detector screen
screen = np.empty(points)
intensity = np.empty(points)
for i in range(points):
    x = W*(i/(points-1)-0.5)
    screen[i] = x
    r = np.sqrt(D**2 + x**2)
    r1 = np.sqrt(D**2 + (x-x1)**2)
    r2 = np.sqrt(D**2 + (x-x2)**2)
    intensity[i] = (1 + np.cos(k*(r1-r2)))/r**2

maxintensity = 0
findingMax = True
doneSearching = False
maxPositions = np.zeros([100,2])
k = 0
dx = []

for i in range(len(intensity)):
    if i == 999: break
    if i >= 2 and (intensity[i] >= intensity[i-1] and intensity[i] >= intensity[i+1]):
        maxPositions[k] = [W*(i/(points-1)-0.5), intensity[i]]
        k += 1

for i in range(k):
    if i == k-1: break
    dx.append(maxPositions[i+1,0] - maxPositions[i,0])

print("dx =", statistics.median(dx))

plt.scatter(maxPositions[:k,0], maxPositions[:k,1])
plt.plot(screen,intensity)
plt.xlabel('x')
plt.ylabel('intensity')
plt.show()