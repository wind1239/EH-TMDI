import numpy as np
from numpy import pi, sqrt
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

freq_ini = 0.5*ws_hz
freq_tar = 1.5*ws_hz

t_end = 300
time = np.linspace(0, t_end, t_end*1e3)

def wo_tmd(x_wo_tmd, t, ms, cs, ks):
    f1 = chirp(t, freq_ini, t_end, freq_tar)
    xs, vs = x_wo_tmd
    return [vs, (f1-cs*vs-ks*xs)/ms]

ans_wo_tmd = odeint(wo_tmd, [0, 0], time, args=(ms, cs, ks))

def tmd(x_tmd, t, ms, cs, ks, mt, c_tmd, k_tmd):
    f1 = chirp(t, freq_ini, t_end, freq_tar)
    xs, vs, xt, vt = x_tmd
    return [vs, (f1-(cs+c_tmd)*vs+c_tmd*vt-(ks+k_tmd)*xs+k_tmd*xt)/ms, vt, (c_tmd*vs-c_tmd*vt+k_tmd*xs-k_tmd*xt)/mt]

ans_tmd = odeint(tmd, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmd, k_tmd))

def tmdi(x_tmdi, t, ms, cs, ks, mt, c_tmd, k_tmd, b):
    f1 = chirp(t, freq_ini, t_end, freq_tar)
    xs, vs, xt, vt = x_tmdi
    return [vs, (f1-(cs+c_tmd)*vs+c_tmd*vt-(ks+k_tmd)*xs+k_tmd*xt)/ms, vt, (c_tmd*vs-c_tmd*vt+k_tmd*xs-k_tmd*xt)/(mt+b)]

ans_tmdi = odeint(tmdi, [0, 0, 0, 0], time, args=(ms, cs, ks, mt, c_tmdi, k_tmdi, b))

plt.figure()
plt.plot(time, ans_wo_tmd[:, 0], label='w/o TMD')
plt.plot(time, ans_tmd[:, 0], label='with TMD')
plt.plot(time, ans_tmdi[:, 0], label='with TMDI')
plt.legend()
plt.grid()
plt.savefig('./fig/TMDI_chirp_force.png')
