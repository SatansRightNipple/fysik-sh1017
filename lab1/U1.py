import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

N = 99
dTheta = np.pi/(2*N)
approx = np.zeros((N,6))
analytical = np.zeros(N)

dt = 0.005 # time step
dt2 = dt/2 # half time step
t = 0 # start time

def rk4(x, v, t):
  xk1 = dt*v
  vk1 = dt*f(x, v, t)
  xk2 = dt*(v+vk1/2)
  vk2 = dt*f(x+xk1/2, v+vk1/2, t+dt/2)
  xk3 = dt*(v+vk2/2)
  vk3 = dt*f(x+xk2/2, v+vk2/2, t+dt/2)
  xk4 = dt*(v+vk3)
  vk4 = dt*f(x+xk3, v+vk3, t+dt)
  x += (xk1+2*xk2+2*xk3+xk4)/6
  v += (vk1+2*vk2+2*vk3+vk4)/6
  t += dt
  return x, v, t

def f(theta, p, t):
  accel = -omega02*np.sin(theta) # pendulum
  #accel += -gamma*p # damping
  #accel += A*np.cos(omega*t) # drive force
  return accel

def step(t, p, theta, position, momentum):
  while True:
    theta, p, t = rk4(theta, p, t)
    # energy
    #H = 0.5*p**2 + 1 - np.cos(theta)
    # position
    position.append(theta)
    momentum.append(p)

    if p >= 0:
        T = t*2
        return T
    
    if theta>np.pi: theta -= 2*np.pi
    if theta<-np.pi: theta += 2*np.pi

for i in range(N):
    theta = i*dTheta # initial angular position
    p = 0. # initial angular velocity
    # model parameters (set m=g=L=1)
    omega0 = 1 # natural frequency
    omega02 = omega0**2
    #gamma = 3/8 # damping coefficient
    #omega = 2/3 # drive frequency
    #A = 1.0 # amplitude of drive force
    position = [] # list to store angular position
    momentum = []

    T = step(t, p, theta, position, momentum)
    
    approx[i,:5] = [theta,2*np.pi,T,2*np.pi*(1+theta**2/16),2*np.pi*(1+theta**2/16+11*theta**4/3072)]
    analytical[i] = np.sqrt(2) * sc.integrate.quad(lambda x: 1/np.sqrt(np.cos(x) - np.cos(theta)), -theta, theta)[0]

approx[:,5] = analytical
plt.plot(approx[1:,0], approx[1:,1])
plt.plot(approx[1:,0], approx[1:,3])
plt.plot(approx[1:,0], approx[1:,4])
plt.plot(approx[1:,0], approx[1:,5])
plt.title("Time vs initial")
plt.xlabel("\u03B8")
plt.ylabel("Time")
plt.scatter(approx[1:,0], approx[1:,2], s=10,color="green")
plt.legend(['Harmonic oscillator', 'Second order powerseries', 'Fourth order powerseries', 'Analytical answer', 'Simulation result'])

plt.show()