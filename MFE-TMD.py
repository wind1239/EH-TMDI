import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
import sympy as sy

def lti_to_sympy(lsys, symplify=True):
    s = sy.Symbol('s')
    G = sy.Poly(lsys.num, s) / sy.Poly(lsys.den, s)
    return sy.simplify(G) if symplify else G

def sympy_to_lti(xpr, s=sy.Symbol('s')):
    num, den = sy.simplify(xpr).as_numer_denom()
    p_num_den = sy.poly(num, s), sy.poly(den, s)
    c_num_den = [sy.expand(p).all_coeffs() for p in p_num_den]
    l_num, l_den = [sy.lambdify((), c)() for c in c_num_den]
    return signal.lti(l_num, l_den)

mu = 0.02
zeta_p = 0.02
zeta_s = 0.01
r_fs = 0.9650
zeta_e = 0.1147
r_fe = 0.9754
u_k = 0.0358

H6 = signal.lti([u_k*r_fs**2, 0, 0], [1, 2*zeta_e*r_fe, r_fe**2])

H2_temp = signal.lti([1, 2*zeta_p+2*zeta_s*r_fs*mu, 1+mu*r_fs**2], [1])
H3_temp = signal.lti([-2*zeta_s*r_fs*mu, -mu*r_fs**2], [1])
H4_temp = signal.lti([-2*zeta_s*r_fs, -r_fs**2], [1])
H5_temp = signal.lti([1, 2*zeta_s*r_fs, r_fs**2], [1])

sH6 = lti_to_sympy(H6)
sH2_temp = lti_to_sympy(H2_temp)
sH3_temp = lti_to_sympy(H3_temp)
sH4_temp = lti_to_sympy(H4_temp)
sH5_temp = lti_to_sympy(H5_temp)

sH2 = sy.simplify(sH2_temp+mu*sH6).expand()
sH3 = sy.simplify(sH3_temp-mu*sH6).expand()
sH4 = sy.simplify(sH4_temp-sH6).expand()
sH5 = sy.simplify(sH5_temp+sH6).expand()

sTF = sy.simplify((1-sH5/sH3)/(sH2*sH5/sH3-sH4)).expand()

TF = sympy_to_lti(sTF)
TF_WO = signal.lti([-1], [1, 2*zeta_p, 1])

r_w, mag, phase = signal.bode(TF, np.arange(0.6, 1.4+1e-3, 1e-3))
r_w_WO, mag_WO, phase_WO = signal.bode(TF_WO, np.arange(0.6, 1.4+1e-3, 1e-3))

plt.figure()
plt.plot(r_w_WO, mag_WO)
plt.plot(r_w, mag)
plt.xlim(0.6, 1.4)
plt.ylim(0, 30)
plt.xlabel('Frequency ratio')
plt.ylabel('Magnitude')
plt.legend(['without TMD', 'with MFE-TMD'])
plt.grid()
plt.savefig('./fig/MFE-TMD.png')
