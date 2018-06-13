import numpy as np
from numpy import pi, sqrt, sin, cos
from scipy.integrate import odeint
from scipy.signal import chirp
import matplotlib.pyplot as plt

ws_hz = 1

mu = 0.01
zeta_s = 0.02
beta = 0.1

f_tmdi = 1/(1+mu+beta)
zeta_tmdi = sqrt( (3*(mu+beta)) / (8*(1+mu+beta)) )

ms = 1.8e4
ws = 2*pi*ws_hz
cs = 2*ms*ws*zeta_s
ks = ms*ws**2

mt = ms*mu

b = ms*beta
w_tmdi = f_tmdi*ws
c_tmdi = 2*(mt+b)*w_tmdi*zeta_tmdi
k_tmdi = (mt+b)*w_tmdi**2

t_end = 10
time = np.linspace(0, t_end, t_end*1e3)

f_wind = 2e3

Ri = 2.7
Re = 3.3
R = Ri+Re
kt = 28e-3
ke = kt
rg = 5
p = 10e-3

ceh = (2*pi/p)**2*((3*ke*kt)/(2*R))*rg

def wo_tmd(x_wo_tmd, t, ms, cs, ks):
    f1 = f_wind*sin(ws*t)
    xs, dxs = x_wo_tmd
    return [dxs, (f1-cs*dxs-ks*xs)/ms]

ans_wo_tmd = odeint(wo_tmd, [0, 0], time, args=(ms, cs, ks))

def EHTMDI1(x_tmdi, t, ms, cs, ks, mt, ct, kt, b):
    f1 = f_wind*sin(ws*t)
    xs, dxs, xt, dxt = x_tmdi
    return [dxs, (f1-(cs+ct)*dxs+ct*dxt-(ks+kt)*xs+kt*xt)/ms, dxt, (ct*dxs-ct*dxt+kt*xs-kt*xt)/(mt+b)]

ans_EHTMDI1 = odeint(EHTMDI1, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmdi, k_tmdi, b))

def EHTMDI2(x_tmdi, t, ms, cs, ks, mt, ct, kt, b, ceh):
    f1 = f_wind*sin(ws*t)
    xs, dxs, xt, dxt = x_tmdi
    return [dxs, (f1-(cs+ct)*dxs+ct*dxt-(ks+kt)*xs+kt*xt)/ms, dxt, (ct*dxs-(ct+ceh)*dxt+kt*xs-kt*xt)/(mt+b)]

ans_EHTMDI2 = odeint(EHTMDI2, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmdi, k_tmdi, b, ceh))

def P_sim(t, dxtdt, R, ke, rg, p):
    W_m = (2*pi/p)*dxtdt*rg
    W_e = W_m*3*8
    i_phase = ke*W_m*cos(W_e*t)/R
    return i_phase**2*Re

plt.figure()
plt.plot(time, ans_wo_tmd[:, 0], label='w/o TMD')
plt.plot(time, ans_EHTMDI1[:, 0], label='with EH-TMDI-1')
plt.xlabel('time (s)')
plt.ylabel('structure disp. (m)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_EHTMDI1_xs.png')

plt.figure()
plt.plot(time, ans_EHTMDI1[:, 3], label='with EH-TMDI-1')
plt.xlabel('time (s)')
plt.ylabel('TMD relative velocity (m/s)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_EHTMDI1_dxt.png')

plt.figure()
plt.plot(time, P_sim(time, ans_EHTMDI1[:, 3], R, ke, rg, p), label='with EH-TMDI-1')
plt.xlabel('time (s)')
plt.ylabel('Power (W)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_EHTMDI1_P.png')

plt.figure()
plt.plot(time, ans_wo_tmd[:, 0], label='w/o TMD')
plt.plot(time, ans_EHTMDI2[:, 0], label='with EH-TMDI-2')
plt.xlabel('time (s)')
plt.ylabel('structure disp. (m)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_EHTMDI2_xs.png')

plt.figure()
plt.plot(time, ans_EHTMDI2[:, 3], label='with EH-TMDI-2')
plt.xlabel('time (s)')
plt.ylabel('TMD relative velocity (m/s)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_EHTMDI2_dxt.png')

plt.figure()
plt.plot(time, P_sim(time, ans_EHTMDI2[:, 3], R, ke, rg, p), label='with EH-TMDI-2')
plt.xlabel('time (s)')
plt.ylabel('Power (W)')
plt.legend()
plt.grid()
plt.savefig('./fig/force_EHTMDI2_P.png')
