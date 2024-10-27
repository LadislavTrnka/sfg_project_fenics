"""
Postprocessing - creating figures, tables, computing epsilon(u)
"""

from dolfin import *
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np

def save_param(name, obj, n_min, n_max, Pressure_min, Pressure_max, norm_eps_lin, norm_eps_non):
    param = {}
    param["Problem"]=obj.problem
    if obj.geometry == "cylinder": #cylinder
        param["geometry"] = "$R$ = {0}, $l$ = {1}".format(1000*obj.R,1000*obj.length)
    elif obj.geometry == "cube": #cube
        param["geometry"] = "$a$ = {0}, $b$ = {1}, $c$ = {2}".format(1000*2*obj.b,1000*2*obj.c,1000*obj.height)
    elif obj.geometry == "beam": # beam
        param["geometry"] = "$a$ = {0}, $b$ = {1}, $c$ = {2}".format(1000*2*obj.b,1000*2*obj.c,1000*obj.length)
    param["dimension"] = obj.W.dim()
    param["YoungR"] = obj.E
    param["PoissonR"] = obj.nu
    param["n_min"] = n_min
    param["n_max"] = n_max
    param["Pressure_min"] = Pressure_min
    param["Pressure_max"] = Pressure_max
    param["norm_eps_lin"] = norm_eps_lin
    param["norm_eps_non"] = norm_eps_non
    with open(name+"/param.txt", 'w') as f:
        f.write(str(param))
    return None

def save_columns(name, *args):
    with open(name, 'w') as f:
        for arg in args:
            f.write(str(arg) +"\n")
    return None

def load_columns(folder):
    Columns = []
    with open(folder+"/data.txt","r") as f:
        for line in f:
            Columns.append(line.strip())
    i = 0
    for Col in Columns:
        Columns[i] = Col.strip('][').split(', ')
        Columns[i][0] = Columns[i][0].replace("'", '')
        for k in range(1, len(Columns[i])):
            Columns[i][k] = float(Columns[i][k])
        i += 1
    return (Columns[0],Columns[1],Columns[2],Columns[3])

def local_project_epsilon_norm(sol, mesh):
    epsilon_u = sym(nabla_grad(sol))
    V = TensorFunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    a_proj = inner(u, v)*dx
    b_proj = inner(epsilon_u, v)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    eps_norm = 0
    dofmap = V.dofmap()
    for cell in cells(mesh):
        eps = []
        for k in dofmap.cell_dofs(cell.index()):
            eps.append(u.vector()[k])
        norm = max(sum(np.abs(eps[0:3])),sum(np.abs(eps[3:6])),sum(np.abs(eps[6:9])))
        if eps_norm<norm:
            eps_norm = norm
    # eps_norm2 = 0
    # element = V.element()
    # for cell in cells(mesh):
    #     x = element.tabulate_dof_coordinates(cell)[0]
    #     eps = u(x)
    #     norm = max(sum(np.abs(eps[0:3])),sum(np.abs(eps[3:6])),sum(np.abs(eps[6:9])))
    #     if eps_norm2<norm:
    #         eps_norm2 = norm
    return eps_norm

def Plot_Ratio(name, data, n_max):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(range(1,n_max+1), data, 'ko')
    plt.axis([1, n_max, 0, 0.6])
    plt.xlabel('n')
    plt.ylabel('Ratio')
    plt.grid(linestyle='dotted')
    major_ticks = np.linspace(0, 0.6, 9)
    minor_ticks = np.linspace(0, 0.6, 17)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(tick.MaxNLocator(integer=True))
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5, linestyle='dotted')
    ax.grid(which='major', alpha=0.5, linestyle='dotted')
    plt.savefig(name)
    #plt.show()
    return None

def Plot_Ratio_Pressure(name, data, Pressure_list):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(Pressure_list, data, 'ko')
    plt.axis([min(Pressure_list), max(Pressure_list), 0, 0.6])
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('Ratio')
    major_ticks = np.linspace(0, 0.6, 9)
    minor_ticks = np.linspace(0, 0.6, 17)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5, linestyle='dotted')
    ax.grid(which='major', alpha=0.5, linestyle='dotted')
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.1e'))
    plt.savefig(name)
    #plt.show()
    return None

def VisualizationVEDO(mesh, u):
    from vedo.dolfin import plot as pl
    from vedo.dolfin import show
    pl(mesh,interactive=False)
    pl(u, mode="displacements",interactive=False)
    show()
    return None

def Make_Plot(name,xlabel,ylabel, *Column):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    (min_y,max_y)=(0,0)
    (min_x,max_x)=(min(Column[0][1:]), max(Column[0][1:]))
    if len(Column)>3:
        for Col in Column[1:]:
            plt.plot(Column[0][1:], Col[1:], 'ko')
            # a = min(Col[1:])
            b = max(Col[1:])
            # if min_y>a:
            #     min_y = a
            if max_y<b:
                max_y=b
    else:
        plt.plot(Column[0][1:], Column[1][1:], 'ko', label='linear model')
        plt.plot(Column[0][1:], Column[2][1:], 'ro', label='nonlinear model')
        # min_y = min(Column[1][1:]+Column[2][1:])
        max_y = max(Column[1][1:]+Column[2][1:])
    plt.axis([min_x,max_x, min_y, max_y])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    major_ticks = np.linspace(min_y,max_y, 11)
    minor_ticks = np.linspace(min_y,max_y, 21)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5, linestyle='dotted')
    ax.grid(which='major', alpha=0.5, linestyle='dotted')
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    if 1/2*(min_x+max_x)>1000 or 1/2*(min_x+max_x)<0.01:
        ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2e'))
    if 1/2*(min_y+max_y)>1000 or 1/2*(min_y+max_y)<0.01:
        ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2e'))
    plt.savefig(name)
    #plt.show()
    return None

def Make_Table(name, *Column):
    from texttable import Texttable
    from tabulate import tabulate
    dig = 3
    #import latextable
    n = len(Column)
    table = Texttable()
    table.set_cols_align(["c"] * n)
    rows = []
    for i in range(0, len(Column[1])):
        pre_row = []
        for Col in Column:
            if isinstance(Col[i], str):
                pre_row.append(Col[i])
            elif Col[i]>1000:
                #pre_row.append(str("{:.2e}".format(Col[i])))
                #pre_row.append(f"{Col[i]:.1E}")
                pre_row.append(Col[i]) 
            else:
                pre_row.append(round(Col[i],dig))
        rows.append(pre_row)
    table.add_rows(rows)
    with open(name, 'w') as f:
        f.write(tabulate(rows, headers="firstrow",numalign="center",stralign="center", tablefmt="latex_raw"))
        #f.write(latextable.draw_latex(table, caption="An example table.", label="table:Neuman"))
    return None

def load_param(names, parameters):
    for folder in names:
        with open(folder+"/param.txt",'r') as inf:
            dict = eval(inf.read())
        parameters[0].append(dict["Problem"])
        parameters[1].append(dict["geometry"])
        parameters[2].append(dict["dimension"])
        parameters[3].append(dict["YoungR"])
        parameters[4].append(dict["PoissonR"])
        parameters[5].append(dict["n_min"])
        parameters[6].append(dict["n_max"])
        parameters[7].append(dict["Pressure_min"])
        parameters[8].append(dict["Pressure_max"])
        parameters[9].append(dict["norm_eps_lin"])
        parameters[10].append(dict["norm_eps_non"])
    return parameters

if __name__ == "__main__": 
    names = ["1_Neumann_n_cylinder","1_Neumann_p_cylinder","2_Mix_n_cube","2_Mix_p_cube","3_Mix_n_plate","3_Mix_p_plate","4_Mix_n_beam","4_Mix_p_beam"]
    # # Make tables from data
    for name in names:
        (Column0, Column1, Column2, Column3) = load_columns(name)
        Make_Table(name+"/table.txt", Column0, Column1, Column2, Column3)
        if "Neumann" in name:
            name_y_ax = "Change of length [mm]"
        elif "cube" in name or "plate" in name:
            name_y_ax = "z-component u(0,0,c) [mm]"
        elif "beam" in name:
            name_y_ax = "the absolute value of x-component u(0,0,l) [mm]"
        if "_n_" in name:
            Plot_Ratio(name+"/fig.pdf",Column3[1:], int(Column0[-1])) 
            Make_Plot(name+"/fig_delta.pdf", "n", name_y_ax, Column0,Column1,Column2)
        if "_p_" in name:
            Plot_Ratio_Pressure(name+"/fig.pdf",Column3[1:], Column0[1:]) 
            Make_Plot(name+"/fig_delta.pdf", "Pressure [Pa]", name_y_ax, Column0,Column1,Column2)
    # # Load parameters of all computations
    parameters = [["Problem"],["Geometry [mm]"],["Dimension"], ["$\YoungR$ [Pa]"], ["$\PoissonR$"], ["$n$ min"], ["$n$ max"], ["Pressure min [Pa]"], ["Pressure max [Pa]"], ["norm eps lin"], ["norm eps non"]]
    parameters = load_param(names, parameters)
    # # # Save parameters
    Make_Table("table_param.txt", parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10])

    