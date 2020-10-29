import networkx as nx
import numpy as np
from scipy import sparse

#-- Network topologies --#
def barab1(n):
    """ Barabasi-Albert preferential attachment. Each node is added with one edge
    Parameter
        n (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    m = 1
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return sparse.dok_matrix(A)

def barab2(n):
    """ Barabasi-Albert preferential attachment. Each node is added with two edges
    Parameter
        n (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    m = 2
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return sparse.dok_matrix(A)

def barab4(n):
    """ Barabasi-Albert preferential attachment. Each node is added with two edges
    Parameter
        n (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    m = 4
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return sparse.dok_matrix(A)

def erdos(mean_degree,n):
    """ Erdos-Renyi random graph.
    Parameter
        mean_degree     (int): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    p = mean_degree/n
    A = nx.adj_matrix(nx.erdos_renyi_graph(n,p)).T
    return sparse.dok_matrix(A)

def random_digraph(mean_degree,n):
    """ Random digraph. Each directed edge is present with probability p = mean_degree/n.
        Since this is a directed graph model, mean_degree = mean in deegree = mean out degree
    Parameter
        mean_degree     (int): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    p = mean_degree/n
    return sparse.random(n,n, density=p, data_rvs=np.ones, format='dok')

def watts2(p,n):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 2
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def watts3(p,n):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 3
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def watts4(p,n):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 4
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def watts5(p,n):
    """ Watts-Strogatz small world model
    Parameter
        p               (float): specific to this topology
        n               (int): n is the size of the network
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    k = 5
    A = nx.adj_matrix(nx.watts_strogatz_graph(n,k,p)).T
    return sparse.dok_matrix(A)

def geom(mean_degree, n):
    """ Random geometric graph
    """
    if n is None:
        n = np.random.randint(smallest_network_size,biggest_network_size)
    r = (mean_degree/(np.pi*n))**.5
    A = nx.adj_matrix(nx.random_geometric_graph(n, r)).T
    return sparse.dok_matrix(A)

def no_edges(n):
    return sparse.csr_matrix((n,n))

def chain(n):
    A = sparse.lil_matrix((n,n))
    for i in range(n - 1):
        A[i+1, i] = 1
    return A

def loop(n):
    A = sparse.lil_matrix((n,n))
    for i in range(n - 1):
        A[i+1, i] = 1
    A[0, -1] = 1
    return A

def ident(n):
    return sparse.eye(n, format="lil")

def remove_edges(A,nedges):
    """ Randomly removes 'nedges' edges from a sparse adj matrix 'A'
    """
    A = A.todok()
    # Remove Edges
    keys = list(A.keys())
    remove_idx = np.random.choice(range(len(keys)),size=nedges, replace=False)
    remove = [keys[i] for i in remove_idx]
    for e in remove:
        A[e] = 0
    return A

def generate_adj(network, n, param=None):
    """ Generate a network with the supplied topology
    Parameters
        network (str)   : one of the options in the list below
            ['barab1', 'barab2', 'barab4', 'erdos', 'random_digraph',
              'watts3', 'watts5', 'watts2','watts4', 'geom', 'no_edges',
              'loop', 'chain', 'ident'
            ]
        param   (float) : specific to the topology
        n       (int)   : size of the topology, optional
    Returns
        An adjacency matrix with the specified network topology
    """
    # the directory function in parameter_experiments.py needs to have the same
    #       network_options as this function, so if more topologies are added, the directory
    #       function in the other file should also be edited
    network_options = ['barab1', 'barab2', 'barab4',
                        'erdos', 'random_digraph',
                        'watts3', 'watts5',
                        'watts2','watts4',
                        'geom', 'no_edges',
                        'loop', 'chain',
                        'ident'
                      ]

    if network not in network_options:
        raise ValueError(f'{network} not in {network_options}')

    if network == 'barab1':
        return barab1(n)
    if network == 'barab2':
        return barab2(n)
    if network == 'barab4':
        return barab4(n)
    if network == 'erdos':
        return erdos(param, n)
    if network == 'random_digraph':
        return random_digraph(param, n)
    if network == 'watts3':
        return watts3(param, n)
    if network == 'watts5':
        return watts5(param, n)
    if network == 'watts2':
        return watts2(param, n)
    if network == 'watts4':
        return watts4(param, n)
    if network == 'geom':
        net = geom(param, n)
    if network == 'no_edges':
        net = no_edges(n)
    if network == 'chain':
        net = chain(n)
    if network == 'loop':
        net = loop(n)
    if network == 'ident':
        net = ident(n)
    return net
