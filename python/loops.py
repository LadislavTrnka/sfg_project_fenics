
# Loops

from dolfin import XDMFFile, info, Constant
from numpy import linspace
import postprocessing as pp

def Pressure_Const(rank, name, obj, n_min, n_max):
    Column0 = ["n"]
    Column1 = ["deltaLIN"]
    Column2 = ["deltaNON"]
    Column3 = ["ratio"] 
    norm_eps_non = []  

    obj.gen_mesh()
    obj.bndry()
    obj.Space()
    info("dim= {}".format(obj.W.dim()))
    obj.construct_LIN_problem()
    u_LIN = obj.solve_LIN_problem()
    delta_LIN = obj.DeltaLIN(u_LIN)
    norm_eps_lin = pp.local_project_epsilon_norm(u_LIN, obj.mesh)
    
    u_LIN_file = XDMFFile(name+"/u_LIN.xdmf")
    u_NON_file = XDMFFile(name+"/u_NON.xdmf")
    u_LIN_file.parameters["flush_output"] = True
    u_NON_file.parameters["flush_output"] = True

    u_LIN.rename("u_LIN","displacement linear")
    u_LIN_file.write(u_LIN,0.0)

    obj.construct_NON_problem()
    try:
        for n in range(n_min, n_max+1):
            info("Parameter n = {0}".format(n))

            obj.n.assign(Constant(n)) 
            u_ = obj.solve_NON_problem()
            u_.rename("u_NON","displacement nonlinear")
            u_NON_file.write(u_,n)

            (delta_NON,ratio) = obj.DeltaNON(delta_LIN, u_)
            Column0.append(n)
            Column1.append(1000*delta_LIN) # meters to millimeters
            Column2.append(1000*delta_NON) # meters to millimeters
            Column3.append(ratio)
            norm_eps_non.append(pp.local_project_epsilon_norm(u_, obj.mesh))

            info("Ration abs(LIN-NON)/LIN =  {0}".format(ratio))
    except:
        info("Newton solver did not converge for n = {0}".format(n))
        n = n-1

    if rank == 0:
        # pp.Plot_Ratio(name+"/fig.pdf",Column3[1:], n) 
        # pp.Make_Plot(name+"/fig_delta.pdf", "n", obj.name_y_ax, Column0,Column1,Column2)
        pp.save_columns(name+"/data.txt", Column0, Column1, Column2, Column3)
        pp.save_param(name, obj, n_min, n, obj.Pressure, obj.Pressure, norm_eps_lin, max(norm_eps_non))
    return None

def n_Const(rank, name, obj, Pressure_min, Pressure_max, number):
    Column0 = ["Pressure"]
    Column1 = ["deltaLIN"]
    Column2 = ["deltaNON"]
    Column3 = ["ratio"]
    norm_eps_lin = []  
    norm_eps_non = []  

    obj.gen_mesh()
    obj.bndry()
    obj.Space()
    info("dim= {}".format(obj.W.dim()))

    u_LIN_file = XDMFFile(name+"/u_LIN.xdmf")
    u_NON_file = XDMFFile(name+"/u_NON.xdmf")
    u_LIN_file.parameters["flush_output"] = True
    u_NON_file.parameters["flush_output"] = True

    i = 1
    Pressure_list = linspace(Pressure_min,Pressure_max, number)
    obj.construct_LIN_problem()
    obj.construct_NON_problem()
    try:
        for Pressure in Pressure_list:
            info("Iteration i = {0}".format(i))
            obj.Pressure  = Pressure
            obj.TractionVect_init()  
          
            u_LIN = obj.solve_LIN_problem()
            delta_LIN = obj.DeltaLIN(u_LIN)
            u_LIN.rename("u_LIN","displacement linear")
            u_LIN_file.write(u_LIN, Pressure)
      
            u_ = obj.solve_NON_problem()
            u_.rename("u_NON","displacement nonlinear")
            u_NON_file.write(u_,Pressure)
            
            (delta_NON,ratio) = obj.DeltaNON(delta_LIN, u_)
            norm_eps_lin.append(pp.local_project_epsilon_norm(u_LIN, obj.mesh))
            norm_eps_non.append(pp.local_project_epsilon_norm(u_, obj.mesh))
            Column0.append(Pressure)
            Column1.append(1000*delta_LIN) # meters to millimeters
            Column2.append(1000*delta_NON) # meters to millimeters
            Column3.append(ratio)
            info("Ration abs(LIN-NON)/LIN =  {0}".format(ratio))
            i +=1
    except:
        info("Newton solver did not converge for pressure = {0}".format(Pressure))
        Pressure = Pressure_list[i-2]

    if rank == 0:
        # pp.Plot_Ratio_Pressure(name+"/fig.pdf",Column3[1:], Column0[1:]) 
        # pp.Make_Plot(name+"/fig_delta.pdf", "Pressure", obj.name_y_ax, Column0,Column1,Column2)
        pp.save_columns(name+"/data.txt", Column0, Column1, Column2, Column3)
        pp.save_param(name, obj, int(obj.n.values()), int(obj.n.values()), Pressure_min, Pressure, max(norm_eps_lin), max(norm_eps_non))
    return None
    