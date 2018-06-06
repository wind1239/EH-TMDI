import numpy as np
from numpy import pi, sqrt
from scipy.integrate import odeint
from scipy.signal import chirp
import matplotlib.pyplot as plt

ws_hz = 1

mu = 0.01
zeta_s = 0.02
beta = 0.1

f_tmd = (1/(1+mu))*sqrt((2-mu)/2)
zeta_tmd = sqrt( (3*mu) / (8*(1+mu)*(1-mu/2)) )

f_tmdi = 1/(1+mu+beta)*sqrt(((1+mu)*(2-mu)-mu*beta)/(2*(1+mu)))
zeta_tmdi = sqrt( (beta**2*mu+6*mu*(1+mu)**2+beta*(1+mu)*(6+7*mu)) / (8*(1+mu)*(1+mu+beta)*(2+mu*(1-beta-mu))) )

ms = 1.8e4
ws = 2*pi*ws_hz
cs = 2*ms*ws*zeta_s
ks = ms*ws**2

mt = ms*mu

w_tmd = f_tmd*ws
c_tmd = 2*mt*w_tmd*zeta_tmd
k_tmd = mt*w_tmd**2

b = ms*beta
w_tmdi = f_tmdi*ws
c_tmdi = 2*(mt+b)*w_tmdi*zeta_tmdi
k_tmdi = (mt+b)*w_tmdi**2

g = 9.81

t_end = 10
time = np.linspace(0, t_end, t_end*1e3)

def wo_tmd(x_wo_tmd, t, ms, cs, ks):
    ddxg = 0.03*g*np.sin(ws*t)
    xs, dxs = x_wo_tmd
    return [dxs, (-ms*ddxg-cs*dxs-ks*xs)/ms]

ans_wo_tmd = odeint(wo_tmd, [0, 0], time, args=(ms, cs, ks))

def tmd(x_tmd, t, ms, cs, ks, mt, ct, kt):
    ddxg = 0.03*g*np.sin(ws*t)
    xs, dxs, xt, dxt = x_tmd
    return [dxs, (-ms*ddxg-(cs+ct)*dxs+ct*dxt-(ks+kt)*xs+kt*xt)/ms, dxt, (-mt*ddxg+ct*dxs-ct*dxt+kt*xs-kt*xt)/mt]

ans_tmd = odeint(tmd, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmd, k_tmd))

def tmdi(x_tmdi, t, ms, cs, ks, mt, ct, kt, b):
    ddxg = 0.03*g*np.sin(ws*t)
    xs, dxs, xt, dxt = x_tmdi
    return [dxs, (-ms*ddxg-(cs+ct)*dxs+ct*dxt-(ks+kt)*xs+kt*xt)/ms, dxt, (-mt*ddxg+ct*dxs-ct*dxt+kt*xs-kt*xt)/(mt+b)]

ans_tmdi = odeint(tmdi, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmdi, k_tmdi, b))

plt.figure()
plt.plot(time, ans_wo_tmd[:, 0], label='w/o TMD')
plt.plot(time, ans_tmd[:, 0], label='with TMD')
plt.plot(time, ans_tmdi[:, 0], label='with TMDI')
plt.xlabel('time (s)')
plt.ylabel('structure disp. (m)')
plt.legend()
plt.grid()
plt.savefig('./fig/base_noEHall_xs.png')

plt.figure()
plt.plot(time, ans_tmd[:, 3], label='with TMD')
plt.plot(time, ans_tmdi[:, 3], label='with TMDI')
plt.xlabel('time (s)')
plt.ylabel('TMD relative velocity (m/s)')
plt.legend()
plt.grid()
plt.savefig('./fig/base_TMDandTMDI_vt.png')
