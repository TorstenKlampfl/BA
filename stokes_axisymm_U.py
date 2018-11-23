from ngsolve import *

# viscosity
nu = 0.001

from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.4), bcs = ("symm", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
mesh = Mesh( geo.GenerateMesh(maxh=0.08))

order = 5
mesh.Curve(order)

Vx = H1(mesh,order=order, dirichlet="wall|cyl|inlet")
Vy = H1(mesh,order=order, dirichlet="symm|wall|cyl|inlet")
Q = H1(mesh,order=order-1)

X = FESpace([Vx,Vy,Q])

Ux,Ur,p = X.TrialFunction()
Vx,Vr,q = X.TestFunction()

r = y
r = IfPos(y-1e-12,r,1e-12)

ux,ur = Ux/r, Ur/r
vx,vr = Vx/r, Vr/r

rdivU = grad(Ux)[0]+grad(Ur)[1]
rdivV = grad(Vx)[0]+grad(Vr)[1]
dudx = 1.0/r*CoefficientFunction((grad(Ux)[0],grad(Ur)[0]))
dudr = 1.0/r*CoefficientFunction((grad(Ux)[1]-Ux/r,grad(Ur)[1]-Ur/r))
dvdx = 1.0/r*CoefficientFunction((grad(Vx)[0],grad(Vr)[0]))
dvdr = 1.0/r*CoefficientFunction((grad(Vx)[1]-Vx/r,grad(Vr)[1]-Vr/r))

stokes = nu* (r*dudr*dvdr+r*dudx*dvdx+1.0/r*ur*vr)+ rdivU*q + rdivV*p - 1e-10*r*p*q
a = BilinearForm(X)
a += SymbolicBFI(stokes,bonus_intorder=0)
a.Assemble()

# nothing here ...
f = LinearForm(X)   
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)

# parabolic inflow at inlet:
uin = CoefficientFunction( (0.4**2 - r**2)/0.4**2)
gfu.components[0].Set(uin*r, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res

velr = CoefficientFunction((gfu.components[0],gfu.components[1]))
invreps = IfPos(r-1e-12,1/r,float("nan"))
vel = CoefficientFunction((invreps*gfu.components[0],invreps*gfu.components[1]))

Draw (velr, mesh, "velocityr", sd=3)
Draw (vel, mesh, "velocity", sd=3)
Draw (Norm(vel), mesh, "norm-vel", sd=3)

print(Integrate(gfu.components[0],mesh,definedon=mesh.Boundaries("inlet")))

print(Integrate(gfu.components[0],mesh,definedon=mesh.Boundaries("outlet")))
