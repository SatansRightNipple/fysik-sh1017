import numpy as np
import matplotlib.pyplot as plt

#wavelength
wavelength = 0.0005 # mm = 5000 Ångström, approx wavelength för green laser pointer
k = 2*np.pi/wavelength

# geometry parameters
D = 2000 # distance to detector screen
W = 20 # width of detector screen
points = 1000 # number of pixels of detector screen
width = np.zeros(4)
width[0] = 20
N = 3
omegaVec = [0.05, 0.15,0.25,0.5]
for b in range(4):
    k = 2*np.pi/wavelength

    d = omegaVec[b]/(N-1) # separation between slits
    pos = np.zeros((N))
    # source positions
    pos = np.linspace(-d/2, d/2, N)
    print(pos)

    # compute time averaged intensity on a detector screen
    screen = np.empty(points)
    intensity = np.empty(points)
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

        intensity[i] = sum/r**2

    minPos = np.zeros([1000,2])
    k = 1
    for i in range(len(intensity)):
        if i == 999: break
        if i >= 2 and (intensity[i] <= intensity[i-1] and intensity[i] <= intensity[i+1]):
            minPos[k] = [W*(i/(points-1)-0.5), intensity[i]]
            k += 1

    for i in range(len(minPos)):
        if i == 99: break
        if i >= 1 and minPos[i+1,0] > 0:
            print("Witdth: ", abs(minPos[i,0]*2))
            width[b] = abs(minPos[i,0]*2)
            break

plt.figure()
plt.plot(screen,intensity)
plt.xlabel('x')
plt.ylabel('intensity')
plt.title("Omega = {}".format(N))

plt.figure()
#plt.scatter([2,3,5,10], [20,13.273273273273274, 7.947947947947949, 4.104104104104104])
plt.scatter([0.05, 0.10,0.25,0.5], width)
print(width)

plt.xlabel('\u03C9')
plt.ylabel('Intensity peak width')
plt.title("Intensity peak width vs width of the slit")
plt.show()