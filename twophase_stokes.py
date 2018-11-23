from ngsolve import *
from netgen.geom2d import SplineGeometry

R=1
nu = CoefficientFunction([1,2])
dom = CoefficientFunction([-1,1])
g = CoefficientFunction([CoefficientFunction((0,-1)),CoefficientFunction((0,-2))])

geo = SplineGeometry()
geo.AddRectangle( (-2*R, -2*R), (2*R, 2*R), leftdomain = 2, rightdomain = 0, bc = "wall")
geo.AddCircle ( (0, 0), r=R, leftdomain=1, rightdomain=2, bc="interface")
geo.SetMaterial(1,"fluid1")
geo.SetMaterial(2,"fluid2")
mesh = Mesh( geo.GenerateMesh(maxh=0.8))

order = 2
mesh.Curve(order+1)


def OuterProduct(a,b):
    return CoefficientFunction( tuple([a[i]*b[j] for i in range(a.dim) for j in range(b.dim)]), dims=(a.dim,b.dim) )

n = specialcf.normal(mesh.dim)
P = Id(mesh.dim) - OuterProduct(n,n)

########### mean curvature field

Vk = VectorH1(mesh, order=order, flags={"definedon": [], "definedonbound": "interface"})

k = GridFunction(Vk)

u,v = Vk.TnT()
ak = BilinearForm(Vk)
ak+=SymbolicBFI(u*v,definedon=mesh.Boundaries("interface"))
ak.Assemble()

fk = LinearForm(Vk)
fk+=SymbolicLFI(InnerProduct(P,grad(v)),definedon=mesh.Boundaries("interface"))
fk.Assemble()

k.vec.data = ak.mat.Inverse(Vk.FreeDofs()) * fk.vec

Draw(k,mesh,"k")

print("mean curv error:", sqrt(Integrate( InnerProduct(k-(mesh.dim-1)/R*n,k-(mesh.dim-1)/R*n),mesh,definedon=mesh.Boundaries("interface"))))

########### stokes field

V1 = VectorH1(mesh, order=order , dirichlet="wall")
V2 = H1(mesh, order=order-1 , dirichlet="", definedon=mesh.Materials("fluid1"))
V3 = H1(mesh, order=order-1 , dirichlet="", definedon=mesh.Materials("fluid2"))
V4 = NumberSpace(mesh)
V = FESpace ([V1,V2,V3,V4])

u, p1, p2, r = V.TrialFunction()
v, q1, q2, s = V.TestFunction()

p=CoefficientFunction([p1,p2])
q=CoefficientFunction([q1,q2])

a = BilinearForm (V)
a += SymbolicBFI ( nu*InnerProduct ( grad(u), grad(v)) )
a += SymbolicBFI ( div(u)*q + div(v)*p )
a += SymbolicBFI ( r*q + s*p )
a.Assemble()

f = LinearForm(V)
f += SymbolicLFI ( g * v)
f += SymbolicLFI ( k * v.Trace(), definedon=mesh.Boundaries("interface"))
f.Assemble()

inv = a.mat.Inverse(V.FreeDofs(),inverse="umfpack")

gfu = GridFunction(V)
gfu.vec.data = inv * f.vec
Draw(gfu.components[0],mesh,"velocity")
Draw(CoefficientFunction([gfu.components[1],gfu.components[2]]),mesh,"pressure")
