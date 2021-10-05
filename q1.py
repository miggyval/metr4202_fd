# /usr/bin/python3

############################
# Imports
############################

# For numerical maths and robotics functions
import numpy as np
from math import *
import modern_robotics as mr

# For plotting and animation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
matplotlib.rcParams['text.usetex'] = True

#######################################
# Pre-defined constants/functions
#######################################

# Define constants
L1 = 1.0        # m
L2 = 1.0        # m
r1 = L1 / 2.0   # m
r2 = L2 / 2.0   # m
m1 = 3.0        # kg
m2 = 2.0        # kg
I1 = 2.0        # kg m^2
I2 = 1.0        # kg m^2

G1 = np.diag([0, 0, I1, m1, m1, m1])
G2 = np.diag([0, 0, I2, m2, m2, m2])

C0 = np.array([
    [ 1, 0, 0, 0 ],
    [ 0, 1, 0, 0 ],
    [ 0, 0, 1, 0 ],
    [ 0, 0, 0, 1 ]
])

C1 = np.array([
    [ 1, 0, 0, L1 ],
    [ 0, 1, 0,  0 ],
    [ 0, 0, 1,  0 ],
    [ 0, 0, 0,  1 ]
])

C2 = np.array([
    [ 1, 0, 0, L1 + L2 ],
    [ 0, 1, 0,       0 ],
    [ 0, 0, 1,       0 ],
    [ 0, 0, 0,       1 ]
])

M0 = np.array([
    [ 1, 0, 0, 0 ],
    [ 0, 1, 0, 0 ],
    [ 0, 0, 1, 0 ],
    [ 0, 0, 0, 1 ]
])

M1 = np.array([
    [ 1, 0, 0, L1 / 2 ],
    [ 0, 1, 0,      0 ],
    [ 0, 0, 1,      0 ],
    [ 0, 0, 0,      1 ]
])

M2 = np.array([
    [ 1, 0, 0, L1 + L2 / 2 ],
    [ 0, 1, 0,           0 ],
    [ 0, 0, 1,           0 ],
    [ 0, 0, 0,           1 ]
])

M3 = np.array([
    [ 1, 0, 0, L1 + L2 ],
    [ 0, 1, 0,       0 ],
    [ 0, 0, 1,       0 ],
    [ 0, 0, 0,       1 ]
])

M01 = mr.TransInv(M0) @ M1
M12 = mr.TransInv(M1) @ M2
M23 = mr.TransInv(M2) @ M3

steps = 900
dt = 1 / 60.0
T = (steps - 1) * dt

S1 = np.array([ 0, 0, 1,   0, 0, 0 ])
S2 = np.array([ 0, 0, 1, 0, -L1, 0 ])
A1 = mr.Adjoint(mr.TransInv(M1)) @ S1
A2 = mr.Adjoint(mr.TransInv(M2)) @ S2

g = np.array([0, -9.8, 0])
Mlist = np.array([M01, M12, M23])
Glist = np.array([G1, G2])
Slist = np.array([S1, S2]).T

# Shorthands for functions
inv = np.linalg.inv
norm = np.linalg.norm

def MassMatrix(thetalist, Mlist, Glist, S):
    n = len(thetalist)
    M = np.zeros((n, n))
    g = [0] * 3
    Ftip = [0] * 6
    dthetalist = [0] * n
    for i in range(n):
        ddthetalist = [0] * n
        ddthetalist[i] = 1
        M[:, i] = mr.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist)
    return M

def ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist):
    Dmat =  MassMatrix(thetalist, Mlist, Glist, Slist)
    clist = mr.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
    glist = mr.GravityForces(thetalist, g, Mlist, Glist, Slist)
    tautiplist = mr.EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)
    ddthetalist = inv(Dmat) @ (taulist - clist - glist - tautiplist)
    return ddthetalist

def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    return (thetalist + dt * np.array(dthetalist), dthetalist + dt * np.array(ddthetalist))

def SimulatePID(theta0list, dtheta0list, thetalistd, dthetalistd, g, Ftipmat, Mlist, Glist, Slist, dt):
    n = len(theta0list)
    N = Ftipmat.shape[0]
    Ftipmat = np.array(Ftipmat).T
    thetamat = np.zeros((N, n))
    dthetamat = np.zeros((N, n))
    dthetamat = np.zeros((N, n))
    thetamat[0, :] = theta0list
    dthetamat[0, :] = dtheta0list
    eint = np.array([0.0, 0.0])
    for i in range(N - 1):
        e = thetalistd - thetamat[i, :]
        eint += e
        de = dthetalistd - dthetamat[i, :]
        taulist = Kp @ e + Ki @ eint * dt + Kd @ (de / dt)
        
        ddthetalist = ForwardDynamics(thetamat[i, :], dthetamat[i, :], taulist, g, Ftipmat[:, i], Mlist, Glist, Slist)
        thetamat[i + 1, :], dthetamat[i + 1, :] = EulerStep(thetamat[i, :], dthetamat[i, :], ddthetalist, dt)
        thetamat[i + 1, :] = -((pi - thetamat[i + 1, :]) % (2.0 * pi) - pi)
        
    return thetamat, dthetamat


##################################
# Initialise simulation variables
################################

# Desired States
thetalistd = np.array([-pi / 3, -pi / 6])
dthetalistd = np.array([0, 0])

# External forces
Ftipmat = np.zeros((steps, 6))

# PID Gain Matrices (Diagonal)
Kp = np.diag([100.0, 0.3])
Ki = np.diag([180.0, 1.0])
Kd = np.diag([0.5, 0.25])

# Initial States
theta0list = np.array([-pi / 4, pi / 4])
dtheta0list = np.array([1.0, -0.5])

# Simulation (returns series of state vectors)
thetamat, dthetamat = SimulatePID(theta0list, dtheta0list, thetalistd, dthetalistd, g, Ftipmat, Mlist, Glist, Slist, dt)

# matplotlib animation function
def animate(i):
    T1 = mr.FKinSpace(C1, Slist[:, :1], thetamat[i, :1])
    T2 = mr.FKinSpace(C2, Slist[:, :2], thetamat[i, :2])
    T1d = mr.FKinSpace(C1, Slist[:, :1], thetalistd[:1])
    T2d = mr.FKinSpace(C2, Slist[:, :2], thetalistd[:2])
    x0 = 0.0
    y0 = 0.0
    x1 = T1[0, 3]
    y1 = T1[1, 3]
    x2 = T2[0, 3]
    y2 = T2[1, 3]
    x1d = T1d[0, 3]
    y1d = T1d[1, 3]
    x2d = T2d[0, 3]
    y2d = T2d[1, 3]
    plt.clf()
    plt.axes(xlim=(-4, 4), ylim=(-3, 3))
    plt.plot([x0, x1d, x2d], [y0, y1d, y2d], 'r--')
    return plt.plot([x0, x1, x2], [y0, y1, y2], 'k')


fig = plt.figure()
FPS = int(1 / dt)
# Might need to change these depending on supported codecs
anim = FuncAnimation(fig, animate, frames=steps, interval=FPS, blit=True)
FFwriter = animation.FFMpegWriter(FPS, extra_args=['-vcodec', 'h264'])
anim.save(str(FPS) + '_fps.mp4', writer=FFwriter)

theta1 = thetamat[:, 0]
theta2 = thetamat[:, 1]
dtheta1 = dthetamat[:, 0]
dtheta2 = dthetamat[:, 1]
timestamp = np.linspace(0, T, steps)

plt.close('all')

plt.subplot(2, 1, 1)
plt.plot(timestamp, theta1, label = r"$\theta_{1}$")
plt.plot(timestamp, theta2, label = r"$\theta_{2}$")
plt.ylim (-pi, pi)
plt.legend(loc = 'lower right')
plt.xlabel("Time")
plt.ylabel("Joint Angles")

plt.subplot(2, 1, 2)
plt.plot(timestamp, dtheta1, label = r"$\dot{\theta}_{1}$")
plt.plot(timestamp, dtheta2, label = r"$\dot{\theta}_{2}$")
plt.ylim (-pi, pi)
plt.legend(loc = 'lower right')
plt.xlabel("Time")
plt.ylabel("Joint Velocities")

plt.show()
