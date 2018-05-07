import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from scipy import signal

zeta_s = 0.02
# zeta_EH = 0.1
# u = 0.01
# b = 0.1

# bu = b+u
# bu1 = 1+b+u
# u1 = 1+u
# zeta_T = sqrt((3*bu)/(8*bu1))
# f = 1/bu1

# TF_WO = signal.lti([-1], [1, 2*zeta_s, 1])
# r_w_WO, mag_WO, phase_WO = signal.bode(TF_WO, np.arange(0.6, 1.4+5e-3, 5e-3))

# TF = signal.lti([b*u,
#                  2*f*bu*(bu*zeta_EH+b*zeta_T),
#                  f**2*bu*(b+4*bu*zeta_EH*zeta_T),
#                  2*f**3*bu**2*zeta_EH],
#                 [b*u,
#                  2*(f*bu**2*zeta_EH+b*u*zeta_s+f*b*u1*bu*zeta_T),
#                  b*(u+f**2*u1*bu)+4*f*bu*(b*zeta_s*zeta_T+bu*zeta_EH*(zeta_s+f*bu1*zeta_T)),
#                  2*f*bu*(b*(f*zeta_s+zeta_T)+bu*zeta_EH*(1+f**2*bu1+4*f*zeta_s*zeta_T)),
#                  f**2*bu*(b+4*bu*zeta_EH*(f*zeta_s+zeta_T)),
#                  2*f**3*bu**2*zeta_EH])
# r_w, mag, phase = signal.bode(TF, np.arange(0.6, 1.4+5e-3, 5e-3))

# plt.figure()
# plt.plot(r_w_WO, mag_WO)
# plt.plot(r_w, mag)
# plt.xlim(0.6, 1.4)
# plt.ylim(0, 30)
# plt.xlabel('Frequency ratio')
# plt.ylabel('Magnitude')
# plt.legend(['without TMD', 'with EH-TMDI'])
# plt.grid()
# plt.savefig('./fig/EH-TMDI.png')

r = np.arange(0.6, 1.4+5e-3, 5e-3)
H = 1/sqrt((1-r**2)**2+(2*zeta_s*r)**2)
plt.figure()
plt.plot(r, H)
plt.show()
