import numpy as np
from numpy import pi, sqrt, sin, cos
from scipy.integrate import odeint
from scipy.signal import chirp
import matplotlib.pyplot as plt

ws_hz = 1

mu = 0.01
zeta_s = 0.02
beta = 0.1

f_tmd = 1/(1+mu)
zeta_tmd = sqrt( (3*mu) / (8*(1+mu)) )

f_tmdi = 1/(1+mu+beta)
zeta_tmdi = sqrt( (3*(mu+beta)) / (8*(1+mu+beta)) )

ms = 1e4
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

t_end = 10
time = np.linspace(0, t_end, t_end*1e3)

f_wind = 1e3

def wo_tmd(x_wo_tmd, t, ms, cs, ks):
    f1 = f_wind*sin(ws*t)
    xs, dxs = x_wo_tmd
    return [dxs, (f1-cs*dxs-ks*xs)/ms]

ans_wo_tmd = odeint(wo_tmd, [0, 0], time, args=(ms, cs, ks))

def tmd(x_tmd, t, ms, cs, ks, mt, c_tmd, k_tmd):
    f1 = f_wind*sin(ws*t)
    xs, dxs, xt, dxt = x_tmd
    return [dxs, (f1-(cs+c_tmd)*dxs+c_tmd*dxt-(ks+k_tmd)*xs+k_tmd*xt)/ms, dxt, (c_tmd*dxs-c_tmd*dxt+k_tmd*xs-k_tmd*xt)/mt]

ans_tmd = odeint(tmd, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmd, k_tmd))

def tmdi(x_tmdi, t, ms, cs, ks, mt, ct, kt, b):
    f1 = f_wind*sin(ws*t)
    xs, dxs, xt, dxt = x_tmdi
    return [dxs, (f1-(cs+ct)*dxs+ct*dxt-(ks+kt)*xs+kt*xt)/ms, dxt, (ct*dxs-ct*dxt+kt*xs-kt*xt)/(mt+b)]

ans_tmdi = odeint(tmdi, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmdi, k_tmdi, b))

plt.figure()
plt.plot(time, ans_wo_tmd[:, 0], label='w/o TMD')
plt.plot(time, ans_tmd[:, 0], label='with TMD')
plt.plot(time, ans_tmdi[:, 0], label='with TMDI')
plt.xlabel('time (s)')
plt.ylabel('structure disp. (m)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_noEHall_xs.png')

plt.figure()
plt.plot(time, ans_tmd[:, 3], label='with TMD')
plt.plot(time, ans_tmdi[:, 3], label='with TMDI')
plt.xlabel('time (s)')
plt.ylabel('TMD relative velocity (m/s)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_TMDandTMDI_dxt.png')
