"""
Elasticity - Density dependent Young modulus
Neumann problem - extension of cylinder
"""

class Neumann_problem():

    # Constants in reference configuration
    E = 20.0e6
    nu = 0.33
    lambda_ = nu*E/(1-nu-2*nu**2)
    mu = E/(2*(1+nu))

    # Mesh geometry
    mesh_density = 60

    problem = "Cylinder"
    geometry = "cylinder"
    length = 0.005
    R = 0.001

    def __init__(self, n, Pressure):
        self.n = Constant(n)
        self.Pressure = Pressure
        self.TractionVect = Constant((0., 0., self.Pressure))
        self.f = Constant((0., 0., 0.))

    def TractionVect_init(self):
        self.TractionVect.assign(Constant((0., 0., self.Pressure)))
        return None

    def gen_mesh(self):
        self.mesh = generate_mesh(Cylinder(Point(0, 0, 0), Point(0, 0, self.length), self.R, self.R), self.mesh_density)        
        return None
        
    def bndry(cls):    
        # Boundaries
        cls.boundary_parts = MeshFunction("size_t", cls.mesh, cls.mesh.topology().dim()-1)
        cls.boundary_parts.set_all(0)

        # Lower side
        class BoundaryL(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], 0, DOLFIN_EPS) 
        bL = BoundaryL()
        bL.mark(cls.boundary_parts, 1)  
        # Upper side
        class BoundaryU(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], cls.length, DOLFIN_EPS)    
        bU = BoundaryU()
        bU.mark(cls.boundary_parts, 2) 
        return None

    def epsilon(self, u):
            return sym(nabla_grad(u))

    def Cauchy_stress(self, u,n):
        return self.lambda_*div(u)*(1-n*div(u))*Identity(3) + 2*self.mu*(1-n*div(u))*self.epsilon(u)

    def Space(self):

       # Define mixed function space
        V = VectorElement("CG", self.mesh.ufl_cell(), 2)
        R = FiniteElement("Real", self.mesh.ufl_cell(),0)
        M = MixedElement([R]*6)
        self.W = FunctionSpace(self.mesh, V*M)
        
        # Define boundary condition
        self.dss = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_parts)

        return None

    def Langrange_multipliers(self,F, u, c, v, d):
        # Lagrange multipliers
        e0 = Constant((1, 0, 0))
        e1 = Constant((0, 1, 0))
        e2 = Constant((0, 0, 1))
        e3 = Expression(( '-x[1]', 'x[0]', '0'), degree=2)
        e4 = Expression(( '-x[2]', '0', 'x[0]'), degree=2)
        e5 = Expression(( '0', '-x[2]', 'x[1]'), degree=2)
        for i, e in enumerate([e0 , e1 , e2 , e3 , e4 , e5]):
            F += c[i]*inner (v, e)*dx + d[i]*inner(u, e)*dx
        return F

    def construct_LIN_problem(self):
        # Weak formulation
        (u, c) = TrialFunctions(self.W)
        (v, d) = TestFunctions(self.W)

        F = inner(self.Cauchy_stress(u,0), self.epsilon(v))*dx - dot(self.f, v)*dx + dot(self.TractionVect, v)*self.dss(1) - dot(self.TractionVect, v)*self.dss(2)
        F = self.Langrange_multipliers(F, u, c, v, d)

        # Construct linear problem
        self.w = Function(self.W)
        problem = LinearVariationalProblem(lhs(F),rhs(F),self.w,None)
        self.solverLIN = LinearVariationalSolver(problem)
        return None

    def solve_LIN_problem(self):
        # Compute solution of linear problem
        self.solverLIN.solve()
        (u,c) = self.w.split() 
        return u

    def construct_NON_problem(self):
        # Construct nonlinear problem
        dw = TrialFunction(self.W)
        (v, d) = TestFunctions(self.W)
        self.w_ = Function(self.W)
        self.w_.assign(self.w)
        (u_, c_) = split(self.w_)

        F_NON = inner(self.Cauchy_stress(u_,self.n), self.epsilon(v))*dx - dot(self.f, v)*dx + dot(self.TractionVect, v)*self.dss(1) - dot(self.TractionVect, v)*self.dss(2)
        F_NON = self.Langrange_multipliers(F_NON, u_, c_, v, d)
       
        J = derivative(F_NON,self.w_,dw)

        problem = NonlinearVariationalProblem(F_NON,self.w_,None,J)
        self.solver = NonlinearVariationalSolver(problem)
        self.solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        self.solver.parameters['newton_solver']['absolute_tolerance'] = 5e-12
        self.solver.parameters['newton_solver']['relative_tolerance'] = 5e-12
        self.solver.parameters['newton_solver']['maximum_iterations'] = 20
        return None

    def solve_NON_problem(self):
        # Compute solution of nonlinear problem
        self.solver.solve()
        (u_,c_) = self.w_.split() 
        return u_

    # Functionals
    def parallel_eval_vec(self, vec, x, component):  
        bb = self.mesh.bounding_box_tree()
        p = Point(x)
        values = np.zeros(self.mesh.topology().dim())
        ic = 0
        cf = bb.compute_first_entity_collision(p)
        inside = cf < self.mesh.num_cells()
        if inside :
            vec.eval_cell(values,x,Cell(self.mesh,cf))
            ic = 1

        comm=MPI.comm_world
        v= MPI.sum(comm, values[component]) / MPI.sum(comm, ic)
        return v

    name_y_ax = "Change of length"
    def DeltaLIN(self, u):
        a = self.parallel_eval_vec(u, [0,0,0], 2)
        b = self.parallel_eval_vec(u, [0,0,self.length], 2)
        return abs(a) + abs(b)

    def DeltaNON(self, delta_LIN, u_):
        a = self.parallel_eval_vec(u_, [0,0,0], 2)
        b = self.parallel_eval_vec(u_, [0,0,self.length], 2)
        delta_NON = abs(a) + abs(b)
        ratio = abs(delta_LIN-delta_NON)/delta_LIN
        return (delta_NON,ratio)

if __name__ == "__main__": 

    from dolfin import *
    from mshr import *
    import numpy as np
    import loops as loop

    # # 1 Neumann boundary conditions - cylinder

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    set_log_level(LogLevel.INFO if rank==0 else LogLevel.INFO)
    parameters["std_out_all_processes"] = False
    parameters["form_compiler"]["quadrature_degree"] = 8

    # Pressure const.
    name = "1_Neumann_n_cylinder"
    n_min = 1
    n_max = 25
    Pressure = 50.0e4
    NP1 = Neumann_problem(n_min, Pressure)
    loop.Pressure_Const(rank, name, NP1, n_min, n_max)

    # n const.
    name = "1_Neumann_p_cylinder"
    n = 4
    Press_min = 100000.
    Press_max = 1000000.
    NP2 = Neumann_problem(n, 0)
    loop.n_Const(rank, name, NP2, Press_min,Press_max,20)

