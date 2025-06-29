#[*TODO list*]
#1. need to decide the data structure of storing flags
#2. then decide the flag combination data structure
#3. implement the code that we could multiply two flags
#4. implement the code that we could multiply two flag combination
#          -- for implementing flag combination, I need to multiply two falg polynomial
import numpy as np
from itertools import permutations
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms import isomorphism as iso



class Graph:
    # write a graph in adjacency matrix, being able to join two graphs together
    adj = None
    labeled_vertices = []
    labeled_name = []
    def __init__(self, size, labeled_vertices, labeled_name):
        # Example: g = Graph(3, [0, 1], ['A', 'B'])
        if(len(labeled_vertices) != len(labeled_name)):
            print("Warning: The size of labeled vertices and labeled name should be the same")
            return
        #self.adj = np.zeros((size, size))
        self.adj = [[False] * size for _ in range(size)]
        self.labeled_vertices = labeled_vertices.copy()
        self.labeled_name = labeled_name.copy()
    
    def __init__(self, adjacency_matrix, labeled_vertices, labeled_name):
        # Example: use the following code to create a graph
        # g = Graph([[1,0],[0,1]], [0, 1], ['A', 'B'])
        if(len(labeled_vertices) != len(labeled_name)): #the size fo labeled vertices and labeled name should be the same
            print("Warning: The size of labeled vertices and labeled name should be the same")
            return
        self.adj = adjacency_matrix.copy()
        self.labeled_vertices = labeled_vertices.copy()
        self.labeled_name = labeled_name.copy()


    #unlabeled isomorphism. Compares [G], [G']
    def __eq__(self, other):
        graph1 = copy.deepcopy(self)
        graph2 = copy.deepcopy(other)
        graph1.remove_isolated_all_vertices()
        graph2.remove_isolated_all_vertices()
        adj1 = graph1.adj
        adj2 = graph2.adj
        G1 = nx.from_numpy_array(adj1)
        G2 = nx.from_numpy_array(adj2)
        return GraphMatcher(G1,G2).is_isomorphic()



    def add_edge(self, i, j):
        self.adj[i][j] = 1
        self.adj[j][i] = 1

    def join_graph(self, graph):
        #multiplies graphs in gluing algebra, single-edge. Removes unlabelled isolated verts
        
        self.remove_isolated_unlab_vertices()
        graph.remove_isolated_unlab_vertices()
        temp_max = len(self.adj) + len(graph.adj)
        # just first copy all the graphs, then we are going to delete the common labeled vertices
        #temp = np.zeros((temp_max, temp_max))
        temp = [[False] * temp_max for _ in range(temp_max)]
        for i in range(len(self.adj)):
            for j in range(len(self.adj)):
                temp[i][j] = self.adj[i][j]
                temp[j][i] = self.adj[j][i]
        for i in range(len(graph.adj)):
            for j in range(len(graph.adj)):
                temp[i+len(self.adj)][j+len(self.adj)] = graph.adj[i][j]
                temp[j+len(self.adj)][i+len(self.adj)] = graph.adj[j][i]
        # delete the common labeled vertices, two vertices are common if they have the same labeled name
        common_labeled_vertices_name = set(self.labeled_name) & set(graph.labeled_name)

        #extract the labeled vertices from the labeled name
        self_common_labeled_vertices = []
        for i in common_labeled_vertices_name:
            self_common_labeled_vertices.append(self.labeled_vertices[self.labeled_name.index(i)])

        graph_common_labeled_vertices_ =[]
        for i in common_labeled_vertices_name:
            graph_common_labeled_vertices_.append(graph.labeled_vertices[graph.labeled_name.index(i)]+len(self.adj))

        # now the graph common labeled vertices are the original index+ the size of the first graph

        # identify the edges of the common labeled vertices
        for i in common_labeled_vertices_name:
            u = self.labeled_vertices[self.labeled_name.index(i)]
            u_prime =  graph.labeled_vertices[graph.labeled_name.index(i)]+len(self.adj)
            for j in range(temp_max):
                #temp[u][j] = temp[u][j] + temp[u_prime][j]
                #temp[j][u] = temp[j][u] + temp[j][u_prime]
                temp[u][j] = int(temp[u][j]) | int(temp[u_prime][j])
                temp[j][u] = int(temp[j][u]) | int(temp[j][u_prime])

        # delete the common labeled vertices of the graph part of the new graph
        temp = np.delete(temp, graph_common_labeled_vertices_, axis = 0)
        temp = np.delete(temp, graph_common_labeled_vertices_, axis = 1)

        # find out the labeled vertices of the new graph
        temp_labeled_vertices = []
        temp_labeled_name = []
        for i in range(len(self.labeled_name)):
            temp_labeled_vertices.append(self.labeled_vertices[i])
            temp_labeled_name.append(self.labeled_name[i])
        
        cnt = 0
        for i in range(len(graph.labeled_name)):
            if graph.labeled_name[i] in common_labeled_vertices_name:
                cnt += 1 # then one vertex is deleted before the index
            if graph.labeled_name[i] not in common_labeled_vertices_name:
                temp_labeled_vertices.append(graph.labeled_vertices[i]+len(self.adj)-cnt)
                temp_labeled_name.append(graph.labeled_name[i])

        # create the new graph
        # the new graph is the union of the two graphs with labeled veritices being the same as the first graph
        # the size is the sum of the two graphs minus the common labeled vertices
        new_graph = Graph(temp, temp_labeled_vertices, temp_labeled_name)
    
        self.adj = temp
        self.labeled_vertices = temp_labeled_vertices
        self.labeled_name = temp_labeled_name

        new_graph.remove_isolated_unlab_vertices()
        return new_graph

    #removes all isolated verts (labelled/unlabelled)
    def remove_isolated_all_vertices(self):
        # Calculate sums of rows and columns to identify vertices with no connections
        self.adj = np.array(self.adj, dtype=int)
        row_sums = self.adj.sum(axis=1)
        col_sums = self.adj.sum(axis=0)
        # Determine all isolated vertices
        isolated = np.where((row_sums + col_sums) == 0)[0]

        # Remove all isolated vertices from the adjacency matrix
        self.adj = np.delete(self.adj, isolated, axis=0)
        self.adj = np.delete(self.adj, isolated, axis=1)

        # Also update labeled vertices and names if applicable
        #NOTE: labels are now wrong, have to adjust
        '''if self.labeled_vertices:
            # Update the labels, ensuring we remove the labels of isolated vertices too
            new_labeled_vertices = []
            new_labeled_names = []
            for index, vertex in enumerate(self.labeled_vertices):
                if vertex not in isolated:
                    new_labeled_vertices.append(vertex)
                    new_labeled_names.append(self.labeled_name[index])

            # Adjust indices of labeled vertices since matrix size has changed
            offset = 0
            for i in range(len(self.adj)):
                while i + offset in isolated:
                    offset += 1
                if i + offset < len(self.adj):
                    self.labeled_vertices[i] = i + offset

            self.labeled_vertices = new_labeled_vertices
            self.labeled_name = new_labeled_names'''

    def print_graph(self):
        # draw the graph using plot

        G = nx.Graph()
        for i in range(len(self.adj)):
            G.add_node(i)
        for i in range(len(self.adj)):
            for j in range(i+1, len(self.adj)):
                if self.adj[i][j] == 1:
                    G.add_edge(i, j)
                    G.add_edge(j, i)
        nx.draw(G, with_labels=True)
        plt.show()
    def copy(self):
        return Graph(self.adj.copy(),self.labeled_vertices.copy(),self.labeled_name.copy())

    #removes unlabelled isolated verts
    def remove_isolated_unlab_vertices(self):
        # Calculate sums of rows and columns to identify vertices with no connections
        self.adj = np.array(self.adj, dtype=int)
        row_sums = self.adj.sum(axis=1)
        col_sums = self.adj.sum(axis=0)
        # Determine all isolated vertices
        isolated = np.where((row_sums + col_sums) == 0)[0]

        # Filter out isolated vertices that are not labeled
        isolated_unlabeled = [idx for idx in isolated if idx not in self.labeled_vertices]

        # Remove isolated, unlabeled vertices from the adjacency matrix
        self.adj = np.delete(self.adj, isolated_unlabeled, axis=0)
        self.adj = np.delete(self.adj, isolated_unlabeled, axis=1)

        # Also update labeled vertices and names if applicable
        if self.labeled_vertices:
            self.labeled_vertices = [v for i, v in enumerate(self.labeled_vertices) if i not in isolated_unlabeled]
            self.labeled_name = [n for i, n in enumerate(self.labeled_name) if i not in isolated_unlabeled]
        #NOTE: at this point, indices of labelled verts could be off. fix later




# Return True iff the graphs represented by adjacency matrices adj1 and adj2 are isomorphic 
# via a mapping that fixes all vertices in `fixed`.
def labeled_isom(graph1, graph2):
    adj1 = copy.deepcopy(graph1.adj)
    adj2 = copy.deepcopy(graph2.adj)
    fixed = copy.deepcopy(graph1.labeled_vertices)

    #setup
    A1 = np.asarray(adj1)
    A2 = np.asarray(adj2)
    assert A1.shape == A2.shape, "Both adjacency matrices must have the same shape"

    G = nx.from_numpy_array(A1)
    H = nx.from_numpy_array(A2)

    fixed_set = set(fixed)

    # Annotate anchors: fixed nodes get their own index, others get None
    for graph in (G, H):
        for v in graph.nodes():
            graph.nodes[v]['anchor'] = v if v in fixed_set else None

    # Only match nodes whose 'anchor' attributes agree
    node_match = iso.categorical_node_match('anchor', None)

    GM = iso.GraphMatcher(G, H, node_match=node_match)
    return GM.is_isomorphic()

#check graph isomorphism between [[G]], [[G']], invariant to isolated all vertices. not used
def unlabeled_isomorphic(input1, input2):
    graph1 = copy.deepcopy(input1)
    graph2 = copy.deepcopy(input2)
    graph1.remove_isolated_all_vertices()
    graph2.remove_isolated_all_vertices()
    # first ensure the degree set\{0} of the two graphs are the same
    degree_set1 = []
    degree_set2 = []
    for i in range(len(graph1.adj)):
        if(sum(graph1.adj[i]) != 0):
            degree_set1.append(sum(graph1.adj[i]))
    for j in range(len(graph2.adj)):
        if(sum(graph2.adj[j]) != 0):
            degree_set2.append(sum(graph2.adj[j]))
    # the degree set excluding 0 should be the same
    degree_set1.sort()
    degree_set2.sort()
    if np.array_equal(degree_set1, degree_set2) == False:
        return 0
    
    # Now we are going to identify the graph without labeled vertices
    f = [] # f[i] is the vertex in graph1 that is identified with vertex i in graph2
    for i in range(0,len(graph2.adj)):
        f.append(-1) # this means the vertex is not identified
    return dfs_identify_graph(graph1, graph2, 0, f)


#Note: SHOULD return whether [[G1]] =isom [[G2]], ignoring labels
def dfs_identify_graph(graph1, graph2, current_vertex_num, f):
    # this function is going to figure out if two graphs are the same, with labeled vertices with the same name identified
    # the two graphs should have the same labeled vertices name
    # the two graphs should have the same size
    
    # however the adjacency matrix of the two graphs could be different
    # we are going to use the adjacency matrix of the first graph as the reference
    # the function is going to return the number of isomorphism between the two graphs
    if (graph1.adj.shape != graph2.adj.shape):
        return 0
    if (current_vertex_num == len(graph1.adj)):# then there is an isomorphism
        return 1
    

    current_degree = sum(graph1.adj[current_vertex_num])
    if current_degree == 0: # any assignment of this vertex would be fine
        return dfs_identify_graph(graph1, graph2, current_vertex_num+1, f)
    
    solution_set = []
    for i in range(len(graph2.adj)):
        if sum(graph2.adj[i]) == current_degree and f[i] == -1:
            solution_set.append(i)
    
    for i in solution_set:
        f[i] = current_vertex_num
        # check if the current assignment is valid
        valid = 1
        for j in range(current_vertex_num):
            # the invariant here is the previously assigned vertices are isomorphic
            # hence only need to check the current vertex and the newly assigned verticesWx
            if(f[j] == -1):
                continue
            if graph1.adj[current_vertex_num][j] != graph2.adj[i][f[j]]:
                valid = 0
                break
        if valid == 0:
            f[i] = -1
            continue
        if dfs_identify_graph(graph1, graph2, current_vertex_num+1, f) == 1:
            return 1
        f[i] = -1
    return 0


class Graph_Combination:
    # Graph_Combination::coefficients[], Graph_Combination::graphs[]
    coefficients = []
    graphs = []

    def __init__(self, coefficients, graphs):
        if len(coefficients) != len(graphs):
            print("Warning: The size of coefficients and graphs should be the same")
            return
        self.coefficients = coefficients.copy()
        temp = []
        for i in range(len(graphs)):
            temp.append(graphs[i].copy())
        self.graphs = temp.copy()

    def __add__(self, other: "Graph_Combination"):
        result = Graph_Combination(self.coefficients, self.graphs)
        for i in range(len(other.graphs)):
            result.graphs.append(other.graphs[i])
            result.coefficients.append(other.coefficients[i])
        result.simplify_expressions()
        return result

    def __eq__(self, other: "Graph_Combination"):
        difference = self + other * (-1)
        return len(difference.coefficients) == 0


    def normalize_graph_size(self): # We are going to make the size of graphs to be the same, meaning adding isolated ulab vertices to small graphs
        max_size = 0
        for i in range(len(self.graphs)):
            max_size = max(max_size, len(self.graphs[i].adj))

        for i in range(len(self.graphs)):
            if len(self.graphs[i].adj) < max_size:
                #temp = np.zeros((max_size, max_size))
                temp = np.zeros((max_size, max_size), dtype=int)

                for j in range(len(self.graphs[i].adj)):
                    for k in range(len(self.graphs[i].adj)):
                        temp[j][k] = self.graphs[i].adj[j][k]
                self.graphs[i].adj = temp
        # Don't need to consider the labeled parts of the graph, since it wouldn't be changed
        
    #combines terms, removes zero coeffs
    def simplify_expressions(self):
        # we are going to simplify the expressions of the graph combination
        # we are going to combine the coefficients of the same graphs

        #combine same graphs
        self.normalize_graph_size()
        for i in range(len(self.graphs)):
            for j in range(i+1, len(self.graphs)):
                if isinstance(self.coefficients[j], cp.Expression):
                    if cp.norm(self.coefficients[j]).value is not None and np.isclose(cp.norm(self.coefficients[j]).value, 0):
                        continue
                elif self.coefficients[j] == 0:
                    continue
                #if identify_same_graph_without_label(self.graphs[i], self.graphs[j]) == 1:
                #if self.graphs[i] == self.graphs[j]: #
                if labeled_isom(self.graphs[i], self.graphs[j]):
                    self.coefficients[i] += self.coefficients[j]
                    self.coefficients[j] = 0

        # remove the zero coefficients
        temp_coefficients = []
        temp_graphs = []
        for i in range(len(self.graphs)):
            if isinstance(self.coefficients[i], cp.Expression):
                if cp.norm(self.coefficients[i]).value is not None and np.isclose(cp.norm(self.coefficients[i]).value, 0):
                    continue
            elif self.coefficients[i] == 0:
                continue
            temp_coefficients.append(self.coefficients[i])
            temp_graphs.append(self.graphs[i])

        self.coefficients = temp_coefficients
        self.graphs = temp_graphs
    
    '''
    def add_graph_combination(self, other):
        for i in range(len(other.graphs)):
            self.graphs.append(other.graphs[i])
            self.coefficients.append(other.coefficients[i])
        self.simplify_expressions()
    '''
    def __mul__(self, coef: float):
        result = Graph_Combination(self.coefficients, self.graphs)
        for i in range(len(result.coefficients)):
            result.coefficients[i] *= coef
        return result

    '''
    def multiply_coefficent(self, coefficient):
        # multiply the coefficient of the graph combination
        for i in range(len(self.coefficients)):
            self.coefficients[i] *= coefficient
    '''

    def print_graph_combination(self):
        # print the graph combination
        for i in range(len(self.graphs)):
            print(self.coefficients[i], end = ' ')
            for j in range(len(self.graphs[i].adj)):
                print(self.graphs[i].adj[j], end = ' ')
            print()
        print()
    
    def copy(self):
        temp_array = []
        for i in range(len(self.coefficients)):
            temp_array.append(self.graphs[i].copy())
        return Graph_Combination(self.coefficients,temp_array)
    
def multiply_graph_combination(graph_combinationa: "Graph_Combination", graph_combinationb: "Graph_Combination"):
        # sum of all pairwise products of terms (with coefficients)
        graph_combination1 = graph_combinationa.copy()
        graph_combination2 = graph_combinationb.copy()
        temp = []
        temp_coefficients = []
        for i in range(len(graph_combination1.graphs)):
            for j in range(len(graph_combination2.graphs)):
                temp_graph = graph_combination1.graphs[i].copy()
                temp.append(temp_graph.join_graph(graph_combination2.graphs[j]))
                temp_coefficients.append(graph_combination1.coefficients[i]*graph_combination2.coefficients[j])
        result = Graph_Combination(temp_coefficients,temp)
        result.simplify_expressions()
        return result

def downward(inp):
    comb = copy.deepcopy(inp)
    for i in range(len(comb.coefficients)):
        comb.graphs[i].labeled_vertices = []
        comb.graphs[i].labeled_name = []
    comb.simplify_expressions()
    comb = comb + Graph_Combination([], [])
    return comb

def quad_form(v, M):
    # the result would be v^T Mv
    result = Graph_Combination([], [])
    for i in range(len(v)):
        for j in range(len(v)):
            temp = multiply_graph_combination(v[i],v[j])
            temp = temp * M[i][j]
            result = result + temp
    return result
    

def Ind(f: Graph_Combination, numVerts) -> Graph_Combination:
    if isinstance(f, Graph):
        print("error, called Ind on a graph, not graph comb")
        return
    result = Graph_Combination([], [])
    for i in range(len(f.coefficients)):
        c_i = f.coefficients[i]
        g_i = f.graphs[i]
        result = result + Ind_graph(g_i, numVerts) * c_i

    return result

def Ind_graph(f: Graph, numVerts) -> Graph_Combination:
    # 1. fatten adj matrix
    f.remove_isolated_unlab_vertices()
    result = Graph_Combination([0], [f])
    adjmat = fattenMatrix(f.adj, numVerts)
    n = len(adjmat)


    # 2. collect all non-edges (i<j)
    extra = [(i,j) 
             for i in range(n) 
             for j in range(i+1, n) 
             if adjmat[i][j] == 0]

    k = len(extra)
    supergraphs = []
    coefs = []

    # 3. for each subset of extra edges
    for r in range(k+1):
        for subset in itertools.combinations(extra, r):
            # 3. copy original
            #B = [row[:] for row in adjmat]    # deep copy
            B = copy.deepcopy(adjmat)
            # add edges in this subset
            for (i,j) in subset:
                B[i][j] = 1
                B[j][i] = 1
            supergraphs.append(B)
            c = len(subset)
            coefs.append((-1)**c)

    #setup the final result
    for i in range(len(supergraphs)):
        matrix = supergraphs[i]
        graph = Graph(matrix, f.labeled_vertices, f.labeled_name)
        result = result + Graph_Combination([coefs[i]], [graph])

    return result



def Hom(f: Graph_Combination, numVerts) -> Graph_Combination:
    if isinstance(f, Graph):
        print("error, called Hom on a graph, not graph comb")
        return
    result = Graph_Combination([], [])
    for i in range(len(f.coefficients)):
        c_i = f.coefficients[i]
        g_i = f.graphs[i]
        result = result + Hom_graph(g_i, numVerts) * c_i

    return result

def Hom_graph(f: Graph, numVerts) -> Graph_Combination:
    # 1. fatten adj matrix
    f.remove_isolated_unlab_vertices()
    result = Graph_Combination([0], [f])
    adjmat = fattenMatrix(f.adj, numVerts)
    n = len(adjmat)


    # 2. collect all non-edges (i<j)
    extra = [(i,j) 
             for i in range(n) 
             for j in range(i+1, n) 
             if adjmat[i][j] == 0]

    k = len(extra)
    supergraphs = []

    # 3. for each subset of extra edges
    for r in range(k+1):
        for subset in itertools.combinations(extra, r):
            # 3. copy original
            #B = [row[:] for row in adjmat]    # deep copy
            B = copy.deepcopy(adjmat)
            # add edges in this subset
            for (i,j) in subset:
                B[i][j] = 1
                B[j][i] = 1
            supergraphs.append(B)

    #setup the final result
    for matrix in supergraphs:
        graph = Graph(matrix, f.labeled_vertices, f.labeled_name)
        result = result + Graph_Combination([1], [graph])

    return result

#if mat is of size nxn for n < numVerts, makes mat of size numVerts x numVerts
#by adding sufficiently many 0's to the right, bottom
def fattenMatrix(mat, numVerts):
    n = len(mat)
    if numVerts > n:
        new_mat = []
        for i in range(numVerts):
            if i < n:
                # extend existing row with zeros
                new_row = np.concatenate((mat[i], np.zeros(numVerts - n, dtype=mat.dtype)))
                new_mat.append(new_row)
            else:
                # entirely new zero row
                new_mat.append([0] * numVerts)
        return new_mat
    else:
        return mat

