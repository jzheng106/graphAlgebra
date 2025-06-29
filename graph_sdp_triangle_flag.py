import graph_multi_ver3 as gm
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import GRAPHS as g


#bucket?
f = gm.Graph_Combination([0.75, -3, 3],[g.empty_graph, g.one_edge_graph, g.two_edge_graph])


#defining the z vectors?
# Assume we now have a vector input
z_0 = [gm.Ind(gm.Graph_Combination([1], [g.one_labeled_empty_graph]), 2),
       gm.Ind(gm.Graph_Combination([1], [g.one_labeled_one_edge_graph]), 2)]
z_gm = [gm.Graph_Combination([1], [g.one_labeled_empty_graph]),
       gm.Graph_Combination([1], [g.one_labeled_one_edge_graph])]

# Define the dimension of the matrix
n = len(z_0)

# Define the variables for the Q matrices 
#Q = cp.Variable((n, n), symmetric=True)
q00 = cp.Variable()
q01 = cp.Variable()
q11 = cp.Variable()

Q = cp.bmat([[q00, q01],
             [q01, q11]])


#setup the quadratic form computation, or =f - z_0^TQz_0
comb_product = gm.quad_form(z_0, Q)
comb_product = comb_product * (-1)
comb_product = comb_product + f

objective = cp.Maximize(comb_product.coefficients[0]) #maximize the coefficient of the empty graph
#####CHANGE HERE#########
constraints = [Q >> 0] 

for i in range(1, len(comb_product.coefficients)):
    constraints.append(comb_product.coefficients[i] == 0)

########Don't change anything below this line ###############################
print("Constraints being applied:")
#for constraint in constraints:
    #print(constraint)

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, eps=1e-10, max_iters=1000000)
print("Optimal value:", problem.value)
print("Q matrix:")
print(Q.value)
print("done")




