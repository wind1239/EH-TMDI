import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

zeta_s_list = [0.02, 0.03, 0.05]
zeta_EH = 0.1
u = 0.01
b = 0.1

bu = b+u
bu1 = 1+b+u
u1 = 1+u

g = np.arange(0.6, 1.4+1e-4, 1e-4)

for zeta_s in zeta_s_list:
    zeta_T = sqrt((3*bu)/(8*bu1))*(1+zeta_s)
    f = (1/bu1)*(1-sqrt(3*bu)*zeta_s)

    if zeta_s == 0.02:
        H = 1/sqrt((1-g**2)**2+(2*zeta_s*g)**2)

        Ns = sqrt((4*f**2*bu**2*((f-g)*(f+g)*bu*zeta_EH-g**2*b*zeta_T)**2+g**2*(b*(-g**2*u+f**2*bu)+4*f**2*bu**2*zeta_EH*zeta_T)**2)/
        (4*(g**2*b*(-(-g**2*u+f**2*bu)*zeta_s+f*bu*(-1+g**2*u1)*zeta_T)-f*bu**2*zeta_EH*(g**2-g**4+f**2*(-1+g**2*bu1)+4*f*g**2*zeta_s*zeta_T))**2
        +g**2*(b*(g**2*u-g**4*u+f**2*bu*(-1+g**2*u1))+4*f*bu*(g**2*b*zeta_s*zeta_T+bu*zeta_EH*((-f**2+g**2)*zeta_s+f*(-1+g**2*bu1)*zeta_T)))**2))

        plt.figure('a')
        plt.plot(g, H, label=r'$N.d.,\zeta_s={0}$'.format(zeta_s))
        plt.plot(g, Ns, label=r'$\zeta_s={0}$'.format(zeta_s))
    
    else:
        Ns = sqrt((4*f**2*bu**2*((f-g)*(f+g)*bu*zeta_EH-g**2*b*zeta_T)**2+g**2*(b*(-g**2*u+f**2*bu)+4*f**2*bu**2*zeta_EH*zeta_T)**2)/
        (4*(g**2*b*(-(-g**2*u+f**2*bu)*zeta_s+f*bu*(-1+g**2*u1)*zeta_T)-f*bu**2*zeta_EH*(g**2-g**4+f**2*(-1+g**2*bu1)+4*f*g**2*zeta_s*zeta_T))**2
        +g**2*(b*(g**2*u-g**4*u+f**2*bu*(-1+g**2*u1))+4*f*bu*(g**2*b*zeta_s*zeta_T+bu*zeta_EH*((-f**2+g**2)*zeta_s+f*(-1+g**2*bu1)*zeta_T)))**2))

        plt.figure('a')
        plt.plot(g, Ns, label=r'$\zeta_s={0}$'.format(zeta_s))

plt.figure('a')
plt.xlim(0.6, 1.4)
plt.ylim(0, 30)
plt.xlabel(r'$g$')
plt.ylabel(r'$N_s(g)$')
plt.legend()
plt.title(r'$\mu = %.2f, \zeta_{EH} = %.2f, \beta = %.2f$'%(u, zeta_EH, b))
plt.grid()
plt.savefig('./fig/in-series_EH-TMDI-zeta_s.png')
