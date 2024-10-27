"""
Elasticity - Density dependent Young modulus
Mixed Boundary Value Problems - Compression of cube/plate
"""

class Mix_problem():

    # Constants in reference configuration
    E = 20.0e6
    nu = 0.33
    lambda_ = nu*E/(1-nu-2*nu**2)
    mu = E/(2*(1+nu))

    # Mesh geometry
    no_elements = 30

    problem = "Cube"
    geometry = "cube" 

    height = 0.005
    b = 0.0025
    c = 0.0025 
    square_under_load = 0.0005

    def __init__(self, n, Pressure):
        self.n =  Constant(n)
        self.Pressure = Pressure
        self.TractionVect = Constant((0., 0., self.Pressure))
        self.f = Constant((0., 0., 0.))

    def TractionVect_init(self):
        self.TractionVect.assign(Constant((0., 0., self.Pressure)))
        return None
    
    def gen_mesh(self):
        self.mesh = BoxMesh(Point(-self.c, -self.b, 0), Point(self.c, self.b, self.height), self.no_elements, self.no_elements, self.no_elements)
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
                return on_boundary and near(x[2], cls.height, DOLFIN_EPS)  and  abs(x[0]) - cls.square_under_load < DOLFIN_EPS and  abs(x[1]) - cls.square_under_load < DOLFIN_EPS
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

    name_y_ax = "z-component u(0,0,c)"
    def DeltaLIN(self, u):
        a = self.parallel_eval_vec(u, [0,0,self.height], 2)
        return abs(a)

    def DeltaNON(self, delta_LIN, u_):
        a = self.parallel_eval_vec(u_, [0,0,self.height], 2)
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

    # # 2 Mixed boundary conditions - half-space (cube)

    # name = "2_Mix_n_cube"
    # n_min = 1
    # n_max = 40
    # Pressure = 10.0e5
    # MPC1 = Mix_problem(n_min, Pressure)
    # loop.Pressure_Const(rank, name, MPC1, n_min, n_max)

    # name = "2_Mix_p_cube"
    # n = 4
    # Press_min = 1.e5
    # Press_max = 14.e5
    # MPC2 = Mix_problem(n, 0)
    # loop.n_Const(rank, name, MPC2, Press_min,Press_max,20)

    # # 3 Mixed boundary conditions - thin plate

    # Pressure const.
    name = "3_Mix_n_plate"
    n_min = 1
    n_max = 40
    Pressure = 17.0e5
    MPP = Mix_problem(n_min, Pressure)
    MPP.problem = "Plate"
    MPP.height = 0.0007
    MPP.b = 0.050
    MPP.c = 0.050
    MPP.square_under_load = 0.020
    loop.Pressure_Const(rank,name, MPP, n_min, n_max)

    # n const.
    name = "3_Mix_p_plate"
    n = 4
    Press_min = 1.e5
    Press_max = 20.e5
    MPP2 = Mix_problem(n, 0)
    MPP2.problem = "Plate"
    MPP2.height = 0.0007
    MPP2.b = 0.050
    MPP2.c = 0.050
    MPP2.square_under_load = 0.020
    loop.n_Const(rank,name, MPP2, Press_min,Press_max, 20)

