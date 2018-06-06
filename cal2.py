import numpy as np
from numpy import pi, sqrt

Ri = 2.7
Re = 6.8
R = Ri+Re
kt = 28e-3
ke = kt
rg = 5
p = 10e-3

ceh = (2*pi/p)**2*((3*ke*kt)/(2*R))*rg

print('C_eh: {0}'.format(ceh))
