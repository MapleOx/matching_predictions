import pandas as pd
import numpy as np
import random
import torch
import gurobipy
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle as pkl
import utils
import algs
from sodapy import Socrata
from datetime import datetime, timedelta


 ## ALGORITHM FUNCTIONS ##

EPS = 1e-5
VERBOSE=False

"""
Generate suggested matching given the first stage graph and a set of possible second-stage graphs.
Assumes the uniform distribution over the second-stage graphs.
Arguments:
    edges1 (list of edges): The first-stage graph
    edges2_list (list of lists of edges): List of the possible second-stage graphs
    weights: Vertex weights on the offline vertices. 
    verbose (bool): True if the solver should print detailed messages. 
Output:
    1. List of vertices that are suggested to match to in the first stage,
    assuming the second-stage graph is generated from the uniform distribution
    over the possible second stages. 
    2. List of edges in the suggested matching. 
"""
def suggested_matching(edges1, edges2_list, weights, verbose=False):
    if verbose:
        print("***Generating suggested matching***")
    num_scenarios = len(edges2_list)
    
    if verbose:
        print("Creating variables and constraints...")
    
    # Define variables
    x = {(i,j): cp.Variable(integer=True) for i,j in edges1}
    y = [{(i,j): cp.Variable() for i,j in edges2} for edges2 in edges2_list]
    
    D1 = set([e[0] for e in edges1])
    D2_list = [set([e[0] for e in edges2]) for edges2 in edges2_list]
    S = set([e[1] for e in edges1] + flatten([[e[1] for e in edges2] for edges2 in edges2_list]))
    
    nbrs1_S = {j: [] for j in S}
    nbrs1_D = {i: [] for i in D1}
    nbrs2_S_list = [{j: [] for j in S} for case in range(num_scenarios)]
    nbrs2_D_list = [{i: [] for i in D2_list[case]} for case in range(num_scenarios)]
    for i,j in edges1:
        nbrs1_S[j].append(i)
        nbrs1_D[i].append(j)
    for case in range(num_scenarios):
        edges2 = edges2_list[case]
        nbrs2_S = nbrs2_S_list[case]
        nbrs2_D = nbrs2_D_list[case]
        for i,j in edges2:
            nbrs2_S[j].append(i)
            nbrs2_D[i].append(j)
#     print(nbrs1_S)
#     print(nbrs1_D)
#     print(nbrs2_S_list)
#     print(nbrs2_D_list)
        
    # Define constraints
    constraints = []
    
    # Non-negativity constraints
    for i,j in edges1:
        constraints += [x[i,j] >= 0]
    for case in range(num_scenarios):
        edges2 = edges2_list[case]
        for i,j in edges2:
            constraints += [y[case][i,j] >= 0]
    
    # Degree constraints
    for j in S:
        for case in range(num_scenarios):
            edges2 = edges2_list[case]
            v1 = cp.sum([x[i,j] for i in nbrs1_S[j]])
            v2 = cp.sum([y[case][i,j] for i in nbrs2_S_list[case][j]])
            if len(nbrs1_S[j]) + len(nbrs2_S_list[case][j]) > 0:
                constraints += [v1 + v2 <= 1]
    for i in D1:
        if len(nbrs1_D[i]) > 0:
            constraints += [cp.sum([x[i,j] for j in nbrs1_D[i]]) <= 1]
    
    for case in range(num_scenarios):
        D2 = D2_list[case]
        for i in D2:
            if len(nbrs2_D_list[case][i]) > 0:
                constraints += [cp.sum([y[case][i,j] for j in nbrs2_D_list[case][i]]) <= 1]
    
    # Objective
    
    # first-stage value
    value1 = cp.sum([weights[j] * cp.sum([x[i,j] for i in nbrs1_S[j]]) for j in S])
    # expected second-stage value
    value2 = 0
#     print("num_scenarios: ", num_scenarios)
    for case in range(num_scenarios):
        value2 += (1/num_scenarios) * cp.sum([weights[j] * cp.sum([y[case][i,j] for i in nbrs2_S_list[case][j]]) for j in S])
    obj = cp.Maximize(value1 + value2)

    if verbose:
        print("Solving integer program...")
    
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose, solver=cp.MOSEK)
    
    # Return suggested offline vertices j
    sugg = []
    for j in S:
        if len(nbrs1_S[j]) > 0 and cp.sum([x[i,j] for i in nbrs1_S[j]]).value == 1:
            sugg.append(j)
            
    if verbose:
        print("Finished generating suggested matching.")
            
        print("Expected value is: ", prob.value)
    
    suggested_matching = []
    for i,j in edges1:
        if x[i,j].value == 1:
            suggested_matching.append((i,j))
        
    return sugg, suggested_matching

"""
Generate the best matching in the first stage graph given a second-stage graph.
Arguments:
    edges1 (list of edges): The first-stage graph
    edges2_list (list of edges): The second-stage graph
    weights: Vertex weights on the offline vertices. 
    verbose (bool): True if the solver should print detailed messages. 
Output:
    1. List of vertices that are suggested to match to in the first stage. 
    2. List of edges in the suggested matching. 
"""
def best_suggested_matching(edges1, edges2, weights, verbose=False):
    if verbose:
        print("***Solving for optimal hindsight matching***")
        print("Creating variables and constraints...")
    D1 = set(e[0] for e in edges1)
    D2 = set(e[0] for e in edges2)
    S = set([e[1] for e in edges1] + [e[1] for e in edges2])
    nbrs1_D = {i: [] for i in D1}
    nbrs2_D = {i: [] for i in D2}
    nbrs1_S = {j: [] for j in S}
    nbrs2_S = {j: [] for j in S}
    for i,j in edges1:
        nbrs1_D[i].append(j)
        nbrs1_S[j].append(i)
    for i,j in edges2:
        nbrs2_D[i].append(j)
        nbrs2_S[j].append(i)
    x = {(i,j): cp.Variable(integer=True) for i,j in edges1}
    y = {(i,j): cp.Variable() for i,j in edges2}

    constraints = []
    for i,j in edges1:
        constraints += [x[i,j] >= 0]
    for i,j in edges2:
        constraints += [y[i,j] >= 0]
        
    for i in D1:
        if len(nbrs1_D[i]) > 0:
            constraints += [cp.sum([x[i,j] for j in nbrs1_D[i]]) <= 1]
    for i in D2:
        if len(nbrs2_D[i]) > 0:
            constraints += [cp.sum([y[i,j] for j in nbrs2_D[i]]) <= 1]
            
    for j in S:
        deg1 = cp.sum([x[i,j] for i in nbrs1_S[j]])
        deg2 = cp.sum([y[i,j] for i in nbrs2_S[j]])
        if len(nbrs1_S[j]) + len(nbrs2_S[j]) > 0:
            constraints += [deg1 + deg2 <= 1]
    
    val1 = cp.sum([weights[j]*cp.sum([x[i,j] for i in nbrs1_S[j]]) for j in S])
    val2 = cp.sum([weights[j]*cp.sum([y[i,j] for i in nbrs2_S[j]]) for j in S])
    obj = cp.Maximize(val1 + val2)
    prob = cp.Problem(obj, constraints)
    
    if verbose:
        print("Solving IP...")
    prob.solve(verbose=verbose, solver=cp.GUROBI)


    # Return suggested offline vertices j
    sugg = []
    for j in S:
        if len(nbrs1_S[j]) > 0 and cp.sum([x[i,j] for i in nbrs1_S[j]]).value == 1:
            sugg.append(j)

    suggested_matching = []
    for i,j in edges1:
        if x[i,j].value == 1:
            suggested_matching.append((i,j))
        
    return sugg, suggested_matching


"""
Implements our algorithm.
Arguments:
    edges1 (list of edges): First stage graph.
    edges2 (list of edges): Second stage graph. 
    sugg: List of vertices that are suggested to match to in the first stage. 
    weights: Vertex weights on the offline vertices.
    R: Target minimum robustness level. 
    verbose (bool): True if the solver should print detailed messages. 
Output:
    Value of the matching returned by our algorithm. 
"""
def our_alg(edges1, edges2, sugg, weights, R, verbose=False):  
    # Solve the first-stage fractional matching using the convex optimization problem
    # maximize sum_j ( w_j * (x_j - F_j(x_j))), s.t. x is a fractional matching
    # Here, 
    #    F_L(x) = 0             if x <= 1-R  and    x - (1-R)(1 + ln(x/(1-R))) if x > 1-R
    #    F_U(x) = -(1-R)ln(1-x) if x <= R    and    -(1-R)ln(1-R) + (x-R)      if x > R  
    
    # Define variables
    D1 = set(e[0] for e in edges1)
    S = set([e[1] for e in edges1] + [e[1] for e in edges2])
    D2 = set(e[0] for e in edges2)

    not_sugg = [j for j in S if j not in sugg]
    
    nbrs_i = {i: [] for i in D1}
    nbrs_j = {j: [] for j in S}
    for i,j in edges1:
        nbrs_i[i].append(j)
        nbrs_j[j].append(i)
    xs = {(i,j): cp.Variable() for i,j in edges1}
    ys = {j: cp.Variable() for j in S}
    zs = {j: cp.Variable() for j in S}
    
#     print(len(S))
#     print(len(weights))
#     print(len(zs))
#     print(len(not_sugg))

    if verbose:
        print("Running our algorithm. The problem has \
        \n\t |D1| = {D1_size}, \
        \n\t |D2| = {D2_size}, \
        \n\t |S|  = {S_size}, \
        \n\t {E1_size} first-stage edges, \
        \n\t {E2_size} second-stage edges.".format(D1_size=len(D1), D2_size=len(D2), S_size=len(S), 
                                                   E1_size=len(edges1), E2_size=len(edges2)))
        
        print("Creating constraints...")
    
    # Define constraints
    constraints = []
    for i,j in edges1:
        constraints += [xs[i,j] >= 0]
    for i in D1:
        if len(nbrs_i[i]) > 0:
            constraints += [cp.sum([xs[i,j] for j in nbrs_i[i]]) <= 1]
    for j in S:
        constraints += [ys[j] == cp.sum([xs[i,j] for i in nbrs_j[j]])]
        constraints += [ys[j] <= 1]
#         if len(nbrs_j[j]) > 0:
#             constraints += [cp.sum([xs[i,j] for i in nbrs_j[j]]) <= 1]
#             constraints += [ys[j] == cp.sum([xs[i,j] for i in nbrs_j[j]])]
#         else:
#             constraints += [ys[j] == 0]
        if j in sugg:
            constraints += [zs[j] >= ys[j]]
            constraints += [zs[j] >= 1-R]
        elif j in not_sugg:
            constraints += [zs[j] >= 0]
            constraints += [zs[j] <= ys[j]]
            constraints += [zs[j] <= R]

    # Define objective
    def hl(x):
        return x - (1 - R)*(1 + cp.log(x/(1-R)))
    
    def hu(x):
        return  (1-R)*cp.log(1-R) - (x - R) - (1-R)*cp.log(1-x)

                
    objective1 = cp.Maximize( cp.sum([weights[j] * (ys[j] - hl(zs[j])) for j in sugg])
                            + cp.sum([weights[j] * (-hu(zs[j])) for j in not_sugg])
                            + EPS * cp.sum([ys[j] for j in S]))
    
    prob1 = cp.Problem(objective1, constraints)
    
#     print(prob1)
    
    if verbose:
        print("Solving the first-stage problem...")
    
    prob1.solve(verbose=verbose, solver=cp.MOSEK)
    
#     print({j: ys[j].value for j in ys})
#     print(objective1.value)
    
    
    if verbose:
        print("Finished solving the first-stage convex program.")
        print("Creating second-stage constraints...")
    
    # Solve for the best second-stage matching subject to first-stage decisions
    # Variables
    x2 = {(i,j): cp.Variable() for i,j in edges2}
    nbrs2_i = {i: [] for i in D2}
    nbrs2_j = {j: [] for j in S}
    
    # Constraints
    constraints2 = []
    for i,j in edges2:
        nbrs2_i[i].append(j)
        nbrs2_j[j].append(i)
        constraints2 += [x2[i,j] >= 0]
        
    # Due to numerical issues, y may slightly exceed 1 or slightly be less than 0.
    # Therefore we project y onto the interval [0, 1]
    for j in S:
        if ys[j].value > 1:
            ys[j].value = 1.0
        elif ys[j].value < 0:
            ys[j].value = 0.0
    
    for j in S:
        if len(nbrs2_j[j]) > 0:
            if ys[j].value > 1:
                print('!!!!! Vertex ', j, '!!!!!')
            constraints2 += [cp.sum([x2[i,j] for i in nbrs2_j[j]]) <= 1 - ys[j].value]
    for i in D2:
        if len(nbrs2_i[i]) > 0:
            constraints2 += [cp.sum([x2[i,j] for j in nbrs2_i[i]]) <= 1]
            
    
     
    # Objective
    objective2 = cp.Maximize(cp.sum([weights[j]*x2[i,j] for i,j in edges2]))
    
    
    
    prob2 = cp.Problem(objective2, constraints2)
#     print(prob2)
    if verbose:
            print("Solving the second-stage problem...")

    prob2.solve(verbose=verbose, solver=cp.MOSEK)
    
    val1 = cp.sum([weights[j] * ys[j] for j in S]).value
    val2 = 0
    val2 = prob2.value

    if verbose:
        print("first stage value", val1)
        print("second stage value", val2)
    return val1 + val2

    
"""
Solves for optimal matching in a bipartite graph.
Arguments:
    edges (list of edges): The graph
    weights (list of floats): Vertex weights on the offline vertices
    verbose (bool): True if the solver should print detailed messages. 
Output:
    value of the maximum-weight matching in the graph. 
"""    
def opt(edges, weights, verbose=False):
    D = set(e[0] for e in edges)
    S = set(e[1] for e in edges)
    nbrs_i = {i: [] for i in D}
    nbrs_j = {j: [] for j in S}
    for i,j in edges:
        nbrs_i[i].append(j)
        nbrs_j[j].append(i)
    x = {(i,j): cp.Variable() for i,j in edges}
#     y = {j: cp.Variable() for j in S}

    constraints = []
    for i,j in edges:
        constraints += [x[i,j] >= 0]
        
    for i in D:
        constraints += [cp.sum([x[i,j] for j in nbrs_i[i]]) <= 1]
    for j in S:
        constraints += [cp.sum([x[i,j] for i in nbrs_j[j]]) <= 1]
#         constraints += [y[j] <= 1]
    
    
    obj = cp.Maximize(cp.sum([weights[j]*cp.sum([x[i,j] for i in nbrs_j[j]]) for j in S]))
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose, solver=cp.MOSEK)
    return prob.value

"""
Solves for optimal matching in a bipartite graph.
Arguments:
    edges1 (list of edges): The first stage graph
    edges2 (list of edges): The second stage graph
    weights (list of floats): Vertex weights on the offline vertices
    verbose (bool): True if the solver should print detailed messages. 
Output:
    value of the maximum-weight matching in the graph. 
"""    
def opt(edges1, edges2, weights, verbose=False):
    
    if verbose:
        print("***Solving for optimal hindsight matching***")
        print("Creating variables and constraints...")
    D1 = set(e[0] for e in edges1)
    D2 = set(e[0] for e in edges2)
    S = set([e[1] for e in edges1] + [e[1] for e in edges2])
    nbrs1_D = {i: [] for i in D1}
    nbrs2_D = {i: [] for i in D2}
    nbrs1_S = {j: [] for j in S}
    nbrs2_S = {j: [] for j in S}
    for i,j in edges1:
        nbrs1_D[i].append(j)
        nbrs1_S[j].append(i)
    for i,j in edges2:
        nbrs2_D[i].append(j)
        nbrs2_S[j].append(i)
    x = {(i,j): cp.Variable() for i,j in edges1}
    y = {(i,j): cp.Variable() for i,j in edges2}

    constraints = []
    for i,j in edges1:
        constraints += [x[i,j] >= 0]
    for i,j in edges2:
        constraints += [y[i,j] >= 0]
        
    for i in D1:
        if len(nbrs1_D[i]) > 0:
            constraints += [cp.sum([x[i,j] for j in nbrs1_D[i]]) <= 1]
    for i in D2:
        if len(nbrs2_D[i]) > 0:
            constraints += [cp.sum([y[i,j] for j in nbrs2_D[i]]) <= 1]
            
    for j in S:
        deg1 = cp.sum([x[i,j] for i in nbrs1_S[j]])
        deg2 = cp.sum([y[i,j] for i in nbrs2_S[j]])
        if len(nbrs1_S[j]) + len(nbrs2_S[j]) > 0:
            constraints += [deg1 + deg2 <= 1]
    
    val1 = cp.sum([weights[j]*cp.sum([x[i,j] for i in nbrs1_S[j]]) for j in S])
    val2 = cp.sum([weights[j]*cp.sum([y[i,j] for i in nbrs2_S[j]]) for j in S])
    obj = cp.Maximize(val1 + val2)
    prob = cp.Problem(obj, constraints)
    
    if verbose:
        print("Solving LP...")
    prob.solve(verbose=verbose, solver=cp.MOSEK)
    
    if verbose:
        print("Optimal hindsight value is: ", prob.value)
    return prob.value


"""
Solves for optimal matching, ASSUMING suggested matching is followed exactly in the first stage.
Arguments:
    edges1 (list of edges): First stage graph.
    edges2 (list of edges): Second stage graph. 
    weights (list of floats): Weights on offline vertices.
    sugg (list of ints): List of vertices that are suggested to match to in the first stage. 
    verbose (bool): True if the solver should print detailed messages. 
Output:
    value of the maximum-weight matching in the graph if the first-stage matching matches exactly the vertices in sugg.
"""  
def advice(edges1, edges2, sugg, weights, verbose=False):
    # Define variables
    D2 = set(e[0] for e in edges2)
    S = set([e[1] for e in edges1] + [e[1] for e in edges2])
    nbrs2_D = {i: [] for i in D2}
    nbrs2_S = {j: [] for j in S}
    for i,j in edges2:
        nbrs2_D[i].append(j)
        nbrs2_S[j].append(i)
    x = {(i,j): cp.Variable() for i,j in edges2}
#     y = {j: cp.Variable() for j in S}

    constraints = []
    for i,j in edges2:
        constraints += [x[i,j] >= 0]
    for j in sugg:
        if len(nbrs2_S[j]) > 0:
            constraints += [cp.sum([x[i,j] for i in nbrs2_S[j]]) == 0]
        
    for i in D2:
        constraints += [cp.sum([x[i,j] for j in nbrs2_D[i]]) <= 1]
    for j in S:
        if len(nbrs2_S[j]) > 0:
            constraints += [cp.sum([x[i,j] for i in nbrs2_S[j]]) <= 1]
#         constraints += [y[j] <= 1]
    
    
    obj = cp.Maximize(cp.sum([weights[j]*cp.sum([x[i,j] for i in nbrs2_S[j]]) for j in S]))
    prob = cp.Problem(obj, constraints)
#     print(prob)
    prob.solve(verbose=verbose, solver=cp.MOSEK)
    return np.sum([weights[j] for j in sugg]) + prob.value


"""
Implements the algorithm that greedily selects the max-weight first-stage matching. 
Arguments:
    edges1 (list of edges): First stage graph.
    edges2 (list of edges): Second stage graph. 
    weights (list of floats): Weights on offline vertices.
    verbose (bool): True if the solver should print detailed messages. 
Output:
    value of the matching returned by the greedy algorithm.
""" 
def greedy(edges1, edges2, weights, verbose=False):
    
    if verbose:
        print("***Running GREEDY algorithm***")
        print("Creating variables and constraints...")
    D1 = set(e[0] for e in edges1)
    D2 = set(e[0] for e in edges2)
    S = set([e[1] for e in edges1] + [e[1] for e in edges2])
    nbrs1_D = {i: [] for i in D1}
    nbrs2_D = {i: [] for i in D2}
    nbrs1_S = {j: [] for j in S}
    nbrs2_S = {j: [] for j in S}
    
    for i,j in edges1:
        nbrs1_D[i].append(j)
        nbrs1_S[j].append(i)
    for i,j in edges2:
        nbrs2_D[i].append(j)
        nbrs2_S[j].append(i)
    x = {(i,j): cp.Variable() for i,j in edges1}
    y = {(i,j): cp.Variable() for i,j in edges2}

    constraints1 = []
    for i,j in edges1:
        constraints1 += [x[i,j] >= 0]
    for i in D1:
        if len(nbrs1_D[i]) > 0:
            constraints1 += [cp.sum([x[i,j] for j in nbrs1_D[i]]) <= 1]
    for j in S:
        if len(nbrs1_S[j]) > 0:
            constraints1 += [cp.sum([x[i,j] for i in nbrs1_S[j]]) <= 1]
            
    obj1 = cp.Maximize(cp.sum([weights[j]*cp.sum([x[i,j] for i in nbrs1_S[j]]) for j in S]))
    prob1 = cp.Problem(obj1, constraints1)
    prob1.solve(verbose=verbose, solver=cp.MOSEK)
        
    constraints2 = []
    for i,j in edges2:
        constraints2 += [y[i,j] >= 0]
        
    for i in D2:
        if len(nbrs2_D[i]) > 0:
            constraints2 += [cp.sum([y[i,j] for j in nbrs2_D[i]]) <= 1]
            
    for j in S:
        amt_prev_filled = 0
        if len(nbrs1_S[j]) > 0:
            amt_prev_filled = cp.sum([x[i,j] for i in nbrs1_S[j]]).value
        if amt_prev_filled > 1:
            amt_prev_filled = 1.0
        elif amt_prev_filled < 0:
            amt_prev_filled = 0.0
        if len(nbrs2_S[j]) > 0:
            constraints2 += [cp.sum([y[i,j] for i in nbrs2_S[j]]) <= 1 - amt_prev_filled]
    
    
    obj2 = cp.Maximize(cp.sum([weights[j]*cp.sum([y[i,j] for i in nbrs2_S[j]]) for j in S]))
    prob2 = cp.Problem(obj2, constraints2)
    
    prob2.solve(verbose=verbose, solver=cp.MOSEK)
    
    return prob1.value + prob2.value


"""
Implements the fully-robust algorithm of Feng et al.
Arguments:
    edges1 (list of edges): First stage graph.
    edges2 (list of edges): Second stage graph. 
    weights: Vertex weights on the offline vertices.
    verbose (bool): True if the solver should print detailed messages. 
Output:
    Value of the matching returned by the fully robust algorithm. 
"""
def fully_robust(edges1, edges2, weights, verbose=False):

    # Solve the first-stage fractional matching using the convex optimization problem
    # maximize sum_j ( w_j * (x_j - F(x_j))), s.t. x is a fractional matching
    # Here, 
    #    F(x) = x^2/2
        
    # Define variables
    D1 = set(e[0] for e in edges1)
    S = set([e[1] for e in edges1] + [e[1] for e in edges2])
    D2 = set(e[0] for e in edges2)

    
    nbrs_i = {i: [] for i in D1}
    nbrs_j = {j: [] for j in S}
    for i,j in edges1:
        nbrs_i[i].append(j)
        nbrs_j[j].append(i)
    xs = {(i,j): cp.Variable() for i,j in edges1}
    ys = {j: cp.Variable() for j in S}
    zs = {j: cp.Variable() for j in S}

    if verbose:
        print("Running the fully robust algorithm. The problem has \
        \n\t |D1| = {D1_size}, \
        \n\t |D2| = {D2_size}, \
        \n\t |S|  = {S_size}, \
        \n\t {E1_size} first-stage edges, \
        \n\t {E2_size} second-stage edges.".format(D1_size=len(D1), D2_size=len(D2), S_size=len(S), 
                                                   E1_size=len(edges1), E2_size=len(edges2)))
        
        print("Creating constraints...")
    
    # Define constraints
    constraints = []
    for i,j in edges1:
        constraints += [xs[i,j] >= 0]
    for i in D1:
        if len(nbrs_i[i]) > 0:
            constraints += [cp.sum([xs[i,j] for j in nbrs_i[i]]) <= 1]
    for j in S:
        constraints += [ys[j] == cp.sum([xs[i,j] for i in nbrs_j[j]])]
        constraints += [ys[j] <= 1]

    # Define objective               
    objective1 = cp.Maximize(cp.sum([weights[j] * (ys[j] - ys[j]**2/2) for j in S]))
    
    prob1 = cp.Problem(objective1, constraints)
    
#     print(prob1)
    
    if verbose:
        print("Solving the first-stage problem...")
    
    prob1.solve(verbose=verbose, solver=cp.MOSEK)
    
#     print({j: ys[j].value for j in ys})
#     print(objective1.value)
    
    
    if verbose:
        print("Finished solving the first-stage convex program.")
        print("Creating second-stage constraints...")
    
    # Solve for the best second-stage matching subject to first-stage decisions
    # Variables
    x2 = {(i,j): cp.Variable() for i,j in edges2}
    nbrs2_i = {i: [] for i in D2}
    nbrs2_j = {j: [] for j in S}
    
    # Constraints
    constraints2 = []
    for i,j in edges2:
        nbrs2_i[i].append(j)
        nbrs2_j[j].append(i)
        constraints2 += [x2[i,j] >= 0]
        
    # Due to numerical issues, y may slightly exceed 1 or slightly be less than 0.
    # Therefore we project y onto the interval [0, 1]
    for j in S:
        if ys[j].value > 1:
            ys[j].value = 1.0
        elif ys[j].value < 0:
            ys[j].value = 0.0
    
    for j in S:
        if len(nbrs2_j[j]) > 0:
            # if ys[j].value > 1:
            #     print('!!!!! Vertex ', j, '!!!!!')
            constraints2 += [cp.sum([x2[i,j] for i in nbrs2_j[j]]) <= 1 - ys[j].value]
    for i in D2:
        if len(nbrs2_i[i]) > 0:
            constraints2 += [cp.sum([x2[i,j] for j in nbrs2_i[i]]) <= 1]
            
    
     
    # Objective
    objective2 = cp.Maximize(cp.sum([weights[j]*x2[i,j] for i,j in edges2]))
    
    
    prob2 = cp.Problem(objective2, constraints2)
#     print(prob2)
    if verbose:
            print("Solving the second-stage problem...")

    prob2.solve(verbose=verbose, solver=cp.MOSEK)
    
    val1 = cp.sum([weights[j] * ys[j] for j in S]).value
    val2 = 0
    val2 = prob2.value

    if verbose:
        print("first stage value", val1)
        print("second stage value", val2)
    return val1 + val2