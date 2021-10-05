# /usr/bin/python3

from os import write
import numpy as np
from math import *
import modern_robotics as mr
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation


def ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist):
    return np.dot(np.linalg.inv(mr.MassMatrix(thetalist, Mlist, Glist, \
                                           Slist)), \
                  np.array(taulist) \
                  - mr.VelQuadraticForces(thetalist, dthetalist, Mlist, \
                                       Glist, Slist) \
                  - mr.GravityForces(thetalist, g, Mlist, Glist, Slist) \
                  - mr.EndEffectorForces(thetalist, Ftip, Mlist, Glist, \
                                      Slist))

def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    return thetalist + dt * np.array(dthetalist), \
           dthetalist + dt * np.array(ddthetalist)

def ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes):
    taumat = np.array(taumat).T
    Ftipmat = np.array(Ftipmat).T
    thetamat = taumat.copy().astype(np.float)
    thetamat[:, 0] = thetalist
    dthetamat = taumat.copy().astype(np.float)
    dthetamat[:, 0] = dthetalist
    for i in range(np.array(taumat).shape[1] - 1):
        for j in range(intRes):
            ddthetalist \
            = ForwardDynamics(thetalist, dthetalist, taumat[:, i], g, \
                              Ftipmat[:, i], Mlist, Glist, Slist)
            thetalist,dthetalist = EulerStep(thetalist, dthetalist, \
                                             ddthetalist, 1.0 * dt / intRes)
        thetamat[:, i + 1] = thetalist
        dthetamat[:, i + 1] = dthetalist
    thetamat = np.array(thetamat).T
    dthetamat = np.array(dthetamat).T
    return thetamat, dthetamat

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

steps = 640
dt = 1 / 30.0
T = (steps - 1) * dt

S1 = np.array([ 0, 0, 1,   0, 0, 0 ])
S2 = np.array([ 0, 0, 1, 0, -L1, 0 ])
A1 = mr.Adjoint(mr.TransInv(M1)) @ S1
A2 = mr.Adjoint(mr.TransInv(M2)) @ S2

thetalist = np.array([0, 0])
dthetalist = np.array([0, 0])
ddthetalist = np.array([0, 0])
g = np.array([0, -9.8, 0])
Ftip = np.array([0, 0, 0, 0, 0, 0])
Mlist = np.array([M01, M12, M23])
Glist = np.array([G1, G2])
Slist = np.array([S1, S2]).T

thetalist = np.array([0.1, 0.1])
dthetalist = np.array([0.1, 0.2])
taumat = np.vstack((
    np.ones((steps // 2, 2)),
    -np.ones((steps // 2, 2))
))

Ftipmat = np.zeros((steps, 6))
intRes = 8

thetamat, dthetamat = ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes)

theta1 = thetamat[:, 0]
theta2 = thetamat[:, 1]
dtheta1 = dthetamat[:, 0]
dtheta2 = dthetamat[:, 1]
timestamp = np.linspace(0, T, steps)

def animate(i):
    T1 = mr.FKinSpace(C1, Slist[:, :1], thetamat[i, :1])
    T2 = mr.FKinSpace(C2, Slist[:, :], thetamat[i, :])
    x0 = 0.0
    y0 = 0.0
    x1 = T1[0, 3]
    y1 = T1[1, 3]
    x2 = T2[0, 3]
    y2 = T2[1, 3]
    plt.clf()
    plt.axes(xlim=(-3, 3), ylim=(-3, 3))
    return plt.plot([x0, x1, x2], [y0, y1, y2])

fig = plt.figure()
anim = FuncAnimation(fig, animate, frames=steps, interval=60, blit=True)
FFwriter = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'h264'])
anim.save('basic_animation.mp4', writer=FFwriter)
# libx264

'''
plt.plot(timestamp, theta1, label = "Theta1")
plt.plot(timestamp, theta2, label = "Theta2")
plt.plot(timestamp, dtheta1, label = "DTheta1")
plt.plot(timestamp, dtheta2, label = "DTheta2")
plt.ylim (-12, 10)
plt.legend(loc = 'lower right')
plt.xlabel("Time")
plt.ylabel("Joint Angles/Velocities")
plt.title("Plot of Joint Angles and Joint Velocities")
plt.show()
'''