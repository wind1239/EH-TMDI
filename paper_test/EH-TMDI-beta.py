import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

zeta_s = 0.02
zeta_EH = 0.1
u = 0.01
b_list = [0.00, 0.05, 0.10, 0.30, 0.50]

g = np.arange(0.6, 1.4+1e-4, 1e-4)

for b in b_list:
    bu = b+u
    bu1 = 1+b+u
    u1 = 1+u
    zeta_T = sqrt((3*bu)/(8*bu1))*(1+zeta_s)
    f = (1/bu1)*(1-sqrt(3*bu)*zeta_s)

    A = f**2-g**2
    B = 2*g*zeta_T*f
    C = (f**2-g**2)*(1-g**2)-u*f**2*g**2-4*zeta_s*zeta_T*f*g**2
    D = 2*g*zeta_T*f*(1-g**2*u1)+2*zeta_s*g*(f**2-g**2)

    if b == 0:
        H = 1/sqrt((1-g**2)**2+(2*zeta_s*g)**2)

        plt.figure('s')
        plt.plot(g, H, label='No device')
        plt.figure('p')
        plt.plot(g, H, label='No device')

    else:
        Ns = sqrt((4*f**2*bu**2*((f-g)*(f+g)*bu*zeta_EH-g**2*b*zeta_T)**2+g**2*(b*(-g**2*u+f**2*bu)+4*f**2*bu**2*zeta_EH*zeta_T)**2)/
        (4*(g**2*b*(-(-g**2*u+f**2*bu)*zeta_s+f*bu*(-1+g**2*u1)*zeta_T)-f*bu**2*zeta_EH*(g**2-g**4+f**2*(-1+g**2*bu1)+4*f*g**2*zeta_s*zeta_T))**2
        +g**2*(b*(g**2*u-g**4*u+f**2*bu*(-1+g**2*u1))+4*f*bu*(g**2*b*zeta_s*zeta_T+bu*zeta_EH*((-f**2+g**2)*zeta_s+f*(-1+g**2*bu1)*zeta_T)))**2))

        RF = sqrt((A**2+B**2)/(C**2+D**2))

        plt.figure('s')
        plt.plot(g, Ns, label=r'$\beta={0}$'.format(b))
        plt.figure('p')
        plt.plot(g, RF, label=r'$\beta={0}$'.format(b))

x_left = 0.6
x_right = 1.4
y_bottom = 0
y_top = 30

plt.figure('s')
plt.xlim(x_left, x_right)
plt.ylim(y_bottom, y_top)
plt.xlabel(r'$g$')
plt.ylabel(r'$N_s(g)$')
plt.legend()
plt.title(r'$\mu = %.2f, \zeta_s = %.2f, \zeta_{EH} = %.2f$'%(u, zeta_s, zeta_EH))
plt.grid()
plt.savefig('./fig/in-series_EH-TMDI-beta.png')

plt.figure('p')
plt.xlim(x_left, x_right)
plt.ylim(y_bottom, y_top)
plt.xlabel(r'$g$')
plt.ylabel(r'$N_s(g)$')
plt.legend()
plt.title(r'$\mu = %.2f, \zeta_s = %.2f, \zeta_{EH} = %.2f$'%(u, zeta_s, zeta_EH))
plt.grid()
plt.savefig('./fig/separated_EH-TMDI-beta.png')
