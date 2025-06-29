import graph_multi_ver3 as gm
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import GRAPHS as g

#SOS VERIFICATION

#bucket?
f = gm.Graph_Combination([1/32],[g.empty_graph])
f = f + gm.Ind(gm.Graph_Combination([1], [g.tpe]), 5) * (-1)

#Hirst matrices

# Y1 = 1/96 * [[4,  -7,  -2,  -5,  4],
#              [-7,  59, -38,  33,  -7],
#              [-2, -38,  44, -18,  -2],
#              [-5,  33, -18,  19,  -5],
#              [4,  -7,  -2,  -5,  4]]
Y1 = (1/96) * np.array([
    [  4,  -7,  -2,  -5,   4],
    [ -7,  59, -38,  33,  -7],
    [ -2, -38,  44, -18,  -2],
    [ -5,  33, -18,  19,  -5],
    [  4,  -7,  -2,  -5,   4]
], dtype=float)

# Y2 = 1/1920 * [[  80,  -275,  -70],
#                [-275, 1632, -446],
#                [ -70,  -446,  748]]
Y2 = (1/1920) * np.array([
    [  80, -275,  -70],
    [-275, 1632, -446],
    [ -70, -446,  748]
], dtype=float)

# Y3 = 1/192 * [[ 32, -43],
#               [-43, 58]]
Y3 = (1/192) * np.array([
    [ 32, -43],
    [-43,  58]
], dtype=float)

# Y4 = 1/960 * [[  65, -214],
#               [-214,  839]]
Y4 = (1/960) * np.array([
    [  65, -214],
    [-214,  839]
], dtype=float)

# Y5 = 1/12 * [[1, 2],
#              [2, 4]]
Y5 = (1/12) * np.array([
    [1, 2],
    [2, 4]
], dtype=float)

# Y6 = 1/120 * [[ 24, -13],
#               [-13,  10]]
Y6 = (1/120) * np.array([
    [ 24, -13],
    [-13,  10]
], dtype=float)



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



test_product6 = gm.quad_form(z_6, Y6)
test_product5 = gm.quad_form(z_5, Y5)

#setup the quadratic form computation, or =f - z_0^TQz_0
comb_product = gm.quad_form(z_1, Y1)
comb_product = comb_product + gm.quad_form(z_2, Y2)
comb_product = comb_product + gm.quad_form(z_3, Y3)
comb_product = comb_product + gm.quad_form(z_4, Y4)
comb_product = comb_product + gm.quad_form(z_5, Y5)
comb_product = comb_product + gm.quad_form(z_6, Y6)

comb_product = gm.downward(comb_product)
comb_product = comb_product + f * (-1)

print (comb_product == gm.Graph_Combination([],[]))



















'''
#########################
##testing remove unlab isolated vertices
# ########################
adj_matrix = [
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0]
]
labeled_vertices = [0] 
labeled_names = [ 'b']
g = gm.Graph(adj_matrix, labeled_vertices, labeled_names)
g.remove_isolated_unlab_vertices()
one_labeled_empty_graph.print_graph()
'''

#########################
#testing Ind, Hom
# #########################
'''
f = gm.Graph_Combination([1, 2, 3], [g.empty_graph, g.one_edge_graph, g.two_edge_graph])
#g = gm.Ind(f, 3)
#g = gm.Hom(g, 3)
h = gm.Hom(f, 3)
h = gm.Ind(h, 3)
'''

