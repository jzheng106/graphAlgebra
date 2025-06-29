import graph_multi_ver3 as gm
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import GRAPHS as g


#bucket?
f = gm.Graph_Combination([1/32],[g.empty_graph])
f = f + gm.Ind(gm.Graph_Combination([1], [g.tpe]), 5) * (-1)


#defining the z vectors?
# Assume we now have a vector input
z_1 = [gm.Ind(gm.Graph_Combination([1], [g.g_31_]), 3),
       gm.Ind(gm.Graph_Combination([1], [g.g_31_02]), 3),
       gm.Ind(gm.Graph_Combination([1], [g.g_31_02_12]), 3),
       gm.Ind(gm.Graph_Combination([1], [g.g_31_01_02]), 3),
       gm.Ind(gm.Graph_Combination([1], [g.g_31_01_02_12]), 3)
]

z_2 = [gm.Ind(gm.Graph_Combination([1, 1], [g.g_43_23, g.g_43_13]), 4),
       gm.Ind(gm.Graph_Combination([1], [g.g_43_13_23]), 4),
       gm.Ind(gm.Graph_Combination([1], [g.g_43_03_13]), 4),
]

z_3 = [gm.Ind(gm.Graph_Combination([1], [g.g_43_12]), 4),
       gm.Ind(gm.Graph_Combination([1, -1, -1], [g.g_43_12_13_23, g.g_43_03_12, g.g_43_03_12_23]), 4),
]

z_4 = [gm.Ind(gm.Graph_Combination([1, -1], [g.g_43_12_23, g.g_43_12_13]), 4),
       gm.Ind(gm.Graph_Combination([1, -1], [g.g_43_03_12_23, g.g_43_03_12_13]), 4),
]

z_5 = [gm.Ind(gm.Graph_Combination([1, -1], [g.g_43_02_12, g.g_43_02_12_13]), 4),
       gm.Ind(gm.Graph_Combination([1, -1], [g.g_43_02_03_12_13, g.g_43_02_03_12_13_23]), 4),
]

z_6 = [gm.Ind(gm.Graph_Combination([1], [g.g_43_02_12_13]), 4),
       gm.Ind(gm.Graph_Combination([1, -1], [g.g_43_02_12_13_23, g.g_43_02_03_12_23]), 4),
]

print("set up vectors")
# Define the variables for the Q matrices 
Y1 = cp.Variable((5,5), symmetric=True)
Y2 = cp.Variable((3,3), symmetric=True)
Y3 = cp.Variable((2,2), symmetric=True)
Y4 = cp.Variable((2,2), symmetric=True)
Y5 = cp.Variable((2,2), symmetric=True)
Y6 = cp.Variable((2,2), symmetric=True)



#setup the quadratic form computation, or =f - z_0^TQz_0
comb_product = f
comb_product = comb_product + gm.quad_form(z_1, Y1) * -1
comb_product = comb_product + gm.quad_form(z_2, Y2) * -1
comb_product = comb_product + gm.quad_form(z_3, Y3) * -1
comb_product = comb_product + gm.quad_form(z_4, Y4) * -1
comb_product = comb_product + gm.quad_form(z_5, Y5) * -1
comb_product = comb_product + gm.quad_form(z_6, Y6) * -1

objective = cp.Maximize(comb_product.coefficients[0]) #maximize the coefficient of the empty graph
#####CHANGE HERE#########
constraints = [Y1 >> 0, Y2 >> 0, Y3 >> 0, Y4 >> 0, Y5 >> 0, Y6 >> 0] 

for i in range(1, len(comb_product.coefficients)):
    constraints.append(comb_product.coefficients[i] == 0)

########Don't change anything below this line ###############################
print("starting sdp computation")

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, eps=1e-4, max_iters=2000)
print("Optimal value:", problem.value)
print("Q matrix:")
print(Y1.value)
print(Y2.value)
print(Y3.value)
print(Y4.value)
print(Y5.value)
print(Y6.value)
print("done")







