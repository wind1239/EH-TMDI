import numpy as np
from numpy import pi, sqrt

Ri = 0.386
Re = 1
R = Ri+Re
kt = 27.6e-3
ke = kt
rg = 10
p = 20e-3

ceh = (2*pi/p)**2*((3*ke*kt)/(2*R))*rg

print('C_eh: {0}'.format(ceh))

ws_hz = 1

mu = 0.01
zeta_s = 0.02
beta = 0.1

# f_tmd = (1/(1+mu))*sqrt((2-mu)/2)
# zeta_tmd = sqrt( (3*mu) / (8*(1+mu)*(1-mu/2)) )

f_tmdi = 1/(1+mu+beta)*sqrt(((1+mu)*(2-mu)-mu*beta)/(2*(1+mu)))
zeta_tmdi = sqrt( (beta**2*mu+6*mu*(1+mu)**2+beta*(1+mu)*(6+7*mu)) / (8*(1+mu)*(1+mu+beta)*(2+mu*(1-beta-mu))) )

ms = 1.8e4
ws = 2*pi*ws_hz
cs = 2*ms*ws*zeta_s
ks = ms*ws**2

mt = ms*mu

# w_tmd = f_tmd*ws
# c_tmd = 2*mt*w_tmd*zeta_tmd
# k_tmd = mt*w_tmd**2

b = ms*beta
w_tmdi = f_tmdi*ws
c_tmdi = 2*(mt+b)*w_tmdi*zeta_tmdi
k_tmdi = (mt+b)*w_tmdi**2

print('C_tmdi: {0}'.format(c_tmdi))
