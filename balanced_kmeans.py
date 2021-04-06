import numpy as np
import scipy.spatial.distance
import random
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding import embed_bqm, unembed_sampleset
from minorminer import find_embedding
from dimod.traversal import connected_components

###
# Davis Arthur
# Auburn University
# 4-5-2021
#
# Balanced k-means clustering: Clusters data set X into k equal sized clusters
# by minimizing the objective function of the generic k-means model.
###

##############
## Examples ##
##############
def test_quantum():
    ''' Example using D-Wave's quantum annealing'''
    X = np.array([[1, 2], [1, 3], [9, 5], [9, 6]])  # input data
    k = 2
    model = genModel(X, k)    # generate BQM model (not yet embedded)
    sampler = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))    # sets D-Wave's sampler, embedding is done automatically
    solution_set = sampler.sample(model, num_reads=100, return_embedding = True)    # run on the D-wave hardware
    print("Embedding: ", solution_set.info["embedding_context"]["embedding"])
    # Count the number of qubits used
    num_qubits = 0
    for entry in solution_set.info["embedding_context"]["embedding"].values():
        num_qubits += len(entry)
    print("Number of qubits: ", num_qubits)
    M, assignments = postprocess(X, solution_set.first.sample)    # postprocess the solution
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

def test_sim():
    ''' Example using D-Wave's simulated annealing'''
    X = np.array([[1, 2], [1, 4], [9, 5], [9, 6]])
    k = 2
    model = genModel(X, k)      # generate BQM model (not yet embedded)
    sample_set = run_sim(model)     # run on simulated solver
    M, assignments = postprocess(X, sample_set.first.sample)    # postprocess the solution
    print("Centroids: ")
    print(M)
    print("Assignments: " + str(assignments))

#################################
## Functions to set up problem ##
#################################
def genModel(X, k, alpha = None, beta = None):
    ''' Generate QUBO model

    Args:
        X: input data as Numpy array
        k: number of clusters

    Returns:
        Binary quadratic model of logical QUBO problem
    '''
    N = np.shape(X)[0]
    return dimod.as_bqm(genA(X, k, alpha = alpha, beta = beta), dimod.BINARY)

def set_sampler():
    ''' Returns D-Wave sampler being used for annealing

    Note: Currently defaults to D-Wave 2000Q_6

    Returns:
        D-Wave sampler
    '''
    return DWaveSampler(solver='DW_2000Q_6')

def get_embedding(sampler, model):
    ''' Find a possible embedding on the hardware

    Args:
        sampler: D-Wave sampler
        model: logical BQM model

    Returns:
        embedding
    '''
    edge_list_model = []
    for key in model.adj.keys():
        for value in model.adj[key].keys():
            if (value, key) not in edge_list_model:
                edge_list_model.append((key, value))
    edge_list_sampler = []
    for key in sampler.adjacency.keys():
        for value in sampler.adjacency[key]:
            if (value, key) not in edge_list_sampler:
                edge_list_sampler.append((key, value))
    return find_embedding(edge_list_model, edge_list_sampler)

def embed(sampler, model, embedding):
    ''' Embeds QUBO on the hardware

    Args:
        sampler: D-Wave sampler being used
        model: QUBO model as BQM
        embedding: embedding returned from get_embedding

    Returns:
        embedded_model: BQM used in run_quantum
    '''
    return embed_bqm(model, embedding, sampler.adjacency)

def run_quantum(sampler, embedded_model, num_reads_in = 100):
    ''' Run the problem on D-Wave hardware
    Args:
        sampler: D-Wave sampler being used to solve the problem
        embedded_model: QUBO model to embed
        num_reads: number of reads during annealing

    Returns:
        solution set to embedded model
    '''
    return sampler.sample(embedded_model, num_reads = num_reads_in, auto_scale = True)

def run_sim(model):
    ''' Run QUBO problem using D-Wave's simulated annealing
    Args:
        model - BQM model to solve
    Returns:
        solution set
    '''
    return dimod.SimulatedAnnealingSampler().sample(model)

def run_exact(model):
    return dimod.ExactSolver().sample(model)

def dwave_postprocess(embedded_solution_set, embedding, model):
    ''' Find logical solution from binary solution of embedded model

    Args:
        embedded_solution: embedded solution set produced by annealing
        embedding: embedding used to convert from logical to embedded model
        model: logical BQM model

    Returns:
        logical solution returned by dwave
    '''
    sample_set = unembed_sampleset(embedded_solution_set, embedding, model)
    return sample_set.first.sample

def postprocess(X, solution):
    ''' Find centroids and assignments from binary solution of logical model

    Args:
        X: input data
        solution: binary solution to logical model

    Returns:
        centroids: Array containing each centnroid as a row vector in a k x d matrix
        assignments: cluster assignments of each point as list
    '''
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    k = len(solution) // N
    num_violations = 0
    assignments = np.array([-1] * N)
    M = np.zeros((k, d))
    cluster_sizes = np.zeros(k)
    for i in range(N):
        cluster_assignment = -1
        num_assigned = 0
        for j in range(k):
            if solution[i + j * N] == 1:
                cluster_assignment = j
                num_assigned += 1
        if num_assigned == 1:
            M[cluster_assignment] += X[i]
            cluster_sizes[cluster_assignment] += 1.0
            assignments[i] = cluster_assignment
        else:
            num_violations += 1
    for i in range(k):
        M[i] /= cluster_sizes[i]
    for i in range(N):
        if assignments[i] == -1:
            min = np.linalg.norm(M[0] - X[i])
            min_index = 0
            for j in range(k):
                new_min = np.linalg.norm(M[j] - X[i])
                if new_min < min:
                    min = new_min
                    min_index = j
            assignments[i] = min_index
            M[min_index] = (M[min_index] * cluster_sizes[min_index] + X[i]) \
                / (cluster_sizes[min_index] + 1)
            cluster_sizes[min_index] += cluster_sizes[min_index] + 1
    return M, assignments, num_violations

######################
## Helper functions ##
######################
def genA(X, k, alpha = None, beta = None):
    ''' Generate QUBO matrix

    Args:
        X: training data
        k: number of clusters

    Returns
        A: numpy array
    '''
    N = np.shape(X)[0]      # number of points
    k = int(k)

    D = genD(X)             # distance matrix
    D /= np.amax(D)

    if alpha == None:
        alpha = 1.0 / (2.0 * N / k - 1.0)
    if beta == None:
        beta = 1.0
    F = genF(N, k)          # column penalty matrix
    return np.kron(np.identity(k), D + alpha * F) + beta * rowpenalty(N, k)

def genF(N, k):
    ''' Generate F matrix

    Args:
        X: training data
        k: number of clusters

    Returns
        F: numpy array
    '''
    return np.ones((N, N)) - 2 * N / k * np.identity(N)

def genG(N, k):
    ''' Generate G matrix

    Args:
        X: training data
        k: number of clusters

    Returns
        G: numpy array
    '''
    return np.ones((k, k)) - 2 * np.identity(k)

def genD(X):
    ''' Generate D matrix

    Args:
        X: training data

    Returns
        D: numpy array
    '''
    D = scipy.spatial.distance.pdist(X, 'sqeuclidean')
    return scipy.spatial.distance.squareform(D)

def genQ(N, k):
    ''' Generate Q matrix

    Args:
        X: training data
        k: number of clusters

    Returns
        Q: numpy array
    '''
    Q = np.zeros((N * k, N * k))
    for i in range(N * k):
        Q[i][N * (i % k) + i // k] = 1.0
    return Q

def rowpenalty(N, k):
    ''' Generate row penalty matrix

    Args:
        X: training data
        k: number of clusters

    Returns
        rowpenalty: numpy array
    '''
    return np.kron(np.ones(k) - 2 * np.identity(k), np.identity(N))
