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

ux,ur,p = X.TrialFunction()
vx,vr,q = X.TestFunction()

r = y
r = IfPos(y-1e-12,r,1e-12)

divu = grad(ux)[0]+grad(ur)[1]+ur/r
divv = grad(vx)[0]+grad(vr)[1]+vr/r
gradu = CoefficientFunction((grad(ux)[0],grad(ux)[1],grad(ur)[0],grad(ur)[1]),dims=(2,2))
gradv = CoefficientFunction((grad(vx)[0],grad(vx)[1],grad(vr)[0],grad(vr)[1]),dims=(2,2))

stokes = nu* (r*InnerProduct(gradu,gradv)+1.0/r*ur*vr)+ r*divu*q+r*divv*p - 1e-10*r*p*q
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
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res

#invreps = IfPos(r-1e-12,1/r,float("nan"))
vel = CoefficientFunction((gfu.components[0],gfu.components[1]))

Draw (r*vel, mesh, "velocityr", sd=3)
Draw (vel, mesh, "velocity", sd=3)
Draw (Norm(vel), mesh, "norm-vel", sd=3)

print(Integrate(r*gfu.components[0],mesh,definedon=mesh.Boundaries("inlet")))

print(Integrate(r*gfu.components[0],mesh,definedon=mesh.Boundaries("outlet")))
