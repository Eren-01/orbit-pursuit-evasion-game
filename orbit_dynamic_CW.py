import numpy as np
from scipy.integrate import solve_ivp

# Definition of parameters#定义一些参数
# Altitudekm = 25508  # 参考卫星高度km
# # mu = 3.98600441 * 100000  # 引力常数
# # Re = 6378145  # m
#
# MIU = 398600.44  # Gravitational parameter of Earth
# RADIUS = 6378145  # # Radius of Earth, m
#
# r = Altitudekm * 1000 + RADIUS  # km
# omega = np.sqrt(MIU / (r * r * r))  # 角速度rad/s
# print("r=", r)
# print("omega=", omega)

# Definition of the equations
# 定义CW方程： 消除z项
# mc = 1
# fx = 0
# fy = 0
# fz = 0
# ax = fx / mc
# ay = fy / mc
# az = fz / mc


class OrbitCalculation:

    def __init__(self, acc, Altitudekm=25508, MIU=398600.44, RADIUS=6378145,dt=0.1):
        self.acc = acc
        self.Altitudekm = Altitudekm  # 参考卫星高度km

        self.MIU = MIU  # Gravitational parameter of Earth
        self.RADIUS = RADIUS  # # Radius of Earth, m
        self.dt = dt
        self.r = self.Altitudekm * 1000 + self.RADIUS  # km
        self.omega = np.sqrt(self.MIU / (self.r * self.r * self.r))  # 角速度rad/s
        # print("r=", self.r)
        # print("omega=", self.omega)

    def cw_equation(self, t, X):
        # 初始参数设置

        ax = self.acc[0]  # 三维机动加速度， m/s
        ay = self.acc[1]
        #az = self.acc[2]

        dX = [0] * 4
        dX[0] = X[2]  # 三维速度
        dX[1] = X[3]


        dX[2] = -2 * self.omega * X[3] + ax  # 三维总体加速度
        dX[3] = 2 * self.omega * X[2] + 3 * self.omega * self.omega * X[1] + ay


        return dX

    def trans_next_state(self, dt,y0):
        t = (0, dt)

        sol = solve_ivp(self.cw_equation, t, y0)
        next_state = sol.y[:, 1]
        return next_state


acc = [0, 0]
oc = OrbitCalculation(acc)
dt=0.1
# y0 = [69.78, 139.56, 104.67, 7.5579, -15.116, 15.116]
y0 = [69.78, 139.56, 7.5579, -15.116]
next_state=oc.trans_next_state(dt,y0)
print(next_state)
p_pos=next_state[0:2]
p_vel=next_state[2:4]
print(p_pos)
print(p_vel)
#
# def cw_equation(t, X):
#     dX = [0] * 6
#
#     dX[0] = X[3]
#     dX[1] = X[4]
#     dX[2] = X[5]  # 速度
#
#     dX[3] = -2 * omega * X[4] + ax
#     dX[4] = 2 * omega * X[3] + 3 * omega * omega * X[1] + ay
#     dX[5] = - omega * omega * X[3] + az
#     return dX


#
#
# # Simulation time 1s
# t = (0, 0.1)
#
# # Initial condition
# # y0 = [None] * 6
# y0 = [69.78, 139.56, 104.67, 7.5579, -15.116, 15.116]
#
# # Solve ODE
# sol = solve_ivp(cw_equation, t, y0)
# print("sol:", sol)
# print(sol.t)
# print(sol.y.shape)
# print("sol.y:", sol.y)
# # import matplotlib.pyplot as plt
# #
# # plt.plot(sol.t, sol.y.T)
# # plt.show()
