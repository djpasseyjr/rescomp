import networkx as nx
import numpy as np
from scipy import sparse

def barabasi(n, mean_degree):
    """ Adjacency matrix for a Barabasi-Albert preferential attachment network.
        Each node is added with `mean_degree` edges.
        Parameters
            mean_degree (int): average edges per node. Must be an even integer
            n (int): Number of nodes in the network
        Returns
            A (csc sparse matrix): Adjacency matrix of the network
    """
    A = nx.adj_matrix(nx.barabasi_albert_graph(n,m)).T
    return A


def erdos(n, mean_degree):
    """ Erdos-Renyi random graph.
        Parameters
            mean_degree (int): average edges per node. Must be an even integer
            n (int): Number of nodes in the network
        Returns
            A (csc sparse matrix): Adjacency matrix of the network
    """
    p = mean_degree/n
    A = nx.adj_matrix(nx.erdos_renyi_graph(n,p)).T
    return A

def random_digraph(n, mean_degree):
    """ Adjacency matrix for a random digraph. Each directed edge is present with probability p = mean_degree/n.
        Since this is a directed graph model, `mean_degree = mean(indegree) = mean(outdegree)`
        Parameters
            mean_degree (int): average edges per node. Must be an even integer
            n (int): Number of nodes in the network
        Returns
            A (csc sparse matrix): Adjacency matrix of the network
    """
    p = mean_degree/n
    return sparse.random(n,n, density=p, data_rvs=np.ones, format='csc')

def watts(mean_degree, n, p=0.1):
    """ Adjacency matrix for a Watts-Strogatz small world network
        Parameters
            mean_degree (int): Average edges per node. Must be an even integer
            n (int): Number of nodes in the network
            param (float): Rewiring probability. p=0 produces a lattice,
                p=1 produces a random graph
        Returns
            A (csc sparse matrix): Adjacency matrix of the network
    """
    if mean_degree % 2 != 0:
        warn("Watts-Strogatz requires a even integer mean degree. Rounding up.")
        mean_degree = 2 * np.ceil(mean_degree / 2)
    A = nx.adj_matrix(nx.watts_strogatz_graph(n, mean_degree, p)).T
    return A

def geometric(mean_degree, n):
    """ Random geometric graph
        Parameters
            mean_degree (int): average edges per node. Must be an even integer
            n (int): Number of nodes in the network
        Returns
            A (csc sparse matrix): Adjacency matrix of the network
    """
    r = (mean_degree/(np.pi*n))**.5
    A = nx.adj_matrix(nx.random_geometric_graph(n, r)).T
    return sparse.dok_matrix(A)
