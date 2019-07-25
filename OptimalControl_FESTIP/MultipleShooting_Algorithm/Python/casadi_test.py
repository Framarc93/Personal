from casadi import *

# Formulate the DAE
x = SX.sym('x', 2)
z = SX.sym('z')
u = SX.sym('u')

f = vertcat(z*x[0]-x[1]+u, x[0])
g = x[1]**2+z-1
h=x[0]**2+x[1]**2+u**2
dae = dict(x = x, p = u, ode= f, z = z, alg = g, quad = h)

# create solver instance
T=10.
N=20
op=dict(t0=0, tf=T/N)
F = integrator('F', 'idas', dae, op)

# empty NLP
w = []
lbw =[]
ubw=[]
G=[]
J=0

# initial conditions
Xk = MX.sym('X0',2)
w+=[Xk]
lbw+=[0,1]
ubw+=[0,1]

for k in range(1, N+1):
    #local control
    Uname='U'+str(k-1)
    Uk = MX.sym(Uname)
    w+=[Uk]
    lbw +=[-1]
    ubw+=[1]

    #call integrator
    Fk = F(x0=Xk, p=Uk)
    J+=Fk['qf']

    #new local state
    Xname = 'X'+str(k)
    Xk=MX.sym(Xname,2)
    w+=[Xk]
    lbw+=[-.25, -inf]
    ubw+=[inf, inf]

    #continuity constraint
    G+=[Fk['xf']-Xk]

#create Nlp solver
nlp=dict(f=J, g=vertcat(*G), x=vertcat(*w))
S=nlpsol('S', 'blocksqp', nlp)

# solve NLP
r=S(lbx=lbw, ubx=ubw, x0=0, lbg=0, ubg=0)

print(r['x'])
