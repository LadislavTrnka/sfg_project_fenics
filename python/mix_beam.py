"""
Elasticity - Density dependent Young modulus
Mixed Boundary Value Problems - bending of beam
"""

class Mix_problem():

    # Constants in reference configuration
    E = 20.0e6
    nu = 0.33
    lambda_ = nu*E/(1-nu-2*nu**2)
    mu = E/(2*(1+nu))

    # Mesh geometry
    mesh_density = 140

    problem = "Beam"
    geometry = "beam" 

    length = 0.010
    b = 0.0005
    c = 0.0005

    def __init__(self, n, Pressure):
        self.n =  Constant(n)
        self.Pressure = Pressure
        self.TractionVect = Constant((self.Pressure, 0., 0.))
        self.f = Constant((0., 0., 0.))

    def TractionVect_init(self):
        self.TractionVect.assign(Constant((self.Pressure, 0., 0.)))
        return None

    def gen_mesh(self):
        self.mesh = generate_mesh(Box(Point(-self.c, -self.b, 0), Point(self.c, self.b, self.length)), self.mesh_density)
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
        # Side under the pressure
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
        self.W = VectorFunctionSpace(self.mesh, 'Lagrange', 2)

        # Define boundary condition
        u_L = Constant((0., 0., 0.))
        self.bcs = DirichletBC(self.W, u_L, self.boundary_parts, 1)
        self.dss = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_parts)
        return None

    def construct_LIN_problem(self):
        # Weak formulation
        self.u = TrialFunction(self.W)
        v = TestFunction(self.W)

        F = inner(self.Cauchy_stress(self.u,0), self.epsilon(v))*dx - dot(self.f, v)*dx + dot(self.TractionVect, v)*self.dss(2)

        # Construct linear problem
        self.u = Function(self.W)
        problem = LinearVariationalProblem(lhs(F),rhs(F),self.u,self.bcs)
        self.solverLIN = LinearVariationalSolver(problem)
        return None

    def solve_LIN_problem(self):
        # Compute solution of linear problem
        self.solverLIN.solve()
        return self.u
    
    def construct_NON_problem(self):
        # Construct nonlinear problem
        du = TrialFunction(self.W)
        v = TestFunction(self.W)
        self.u_ = Function(self.W)
        self.u_.assign(self.u)

        F_NON = inner(self.Cauchy_stress(self.u_,self.n), self.epsilon(v))*dx - dot(self.f, v)*dx + dot(self.TractionVect, v)*self.dss(2)
        J = derivative(F_NON,self.u_,du)

        problem = NonlinearVariationalProblem(F_NON,self.u_,self.bcs,J)
        self.solver = NonlinearVariationalSolver(problem)
        self.solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        self.solver.parameters['newton_solver']['absolute_tolerance'] = 5e-12
        self.solver.parameters['newton_solver']['relative_tolerance'] = 5e-12
        self.solver.parameters['newton_solver']['maximum_iterations'] = 20
        return None

    def solve_NON_problem(self):
        # Compute solution of nonlinear problem
        self.solver.solve()
        return self.u_

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

    name_y_ax = "the absolute value of x-component u(0,0,l)"
    def DeltaLIN(self, u):
        a = self.parallel_eval_vec(u, [0,0,self.length], 0)
        return abs(a)

    def DeltaNON(self, delta_LIN, u_):
        a = self.parallel_eval_vec(u_, [0,0,self.length], 0)
        delta_NON = abs(a)
        ratio = abs(delta_LIN-delta_NON)/delta_LIN
        return (delta_NON,ratio)

if __name__ == "__main__": 

    from dolfin import *
    from mshr import *
    import numpy as np
    import loops as loop

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    set_log_level(LogLevel.INFO if rank==0 else LogLevel.INFO)
    parameters["std_out_all_processes"] = False
    parameters["form_compiler"]["quadrature_degree"] = 8

    # # 4 Mixed boundary conditions - bending

    # Pressure const.
    name = "4_Mix_n_beam"
    n_min = 1
    n_max = 10
    Pressure = 20.0e3
    MP1 = Mix_problem(n_min, Pressure)
    loop.Pressure_Const(rank,name, MP1, n_min, n_max)

    # n const.
    name = "4_Mix_p_beam"
    n = 4
    Press_min = 1.0e3
    Press_max = 30.0e3
    MP2 = Mix_problem(n, 0)
    loop.n_Const(rank, name, MP2, Press_min,Press_max,20)

