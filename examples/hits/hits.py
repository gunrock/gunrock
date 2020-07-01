import networkx as nx
import numpy as np
import sys
from scipy.io import mmread
from scipy.sparse import coo_matrix
np.set_printoptions(threshold=sys.maxsize)

def hits(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    max_iter : integer, optional
      Maximum number of iterations in power method.

    tol : float, optional
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of each node for power method iteration.

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> h,a=nx.hits(G)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("hits() not defined for graphs with multiedges.")
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        # normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        hlast = h
        # print("NX Hlast")
        # print(hlast)
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get('weight', 1)
        # print("NX Auths")
        # print(a)
        # now multiply h=Ga
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get('weight', 1)
        # print("NX Hubs")
        # print(h)
        # normalize vector
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        # normalize vector
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        # check convergence, l1 norm
        err = sum([abs(h[n] - hlast[n]) for n in h])
        if err < tol:
            break
    else:
        pass
        # raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a

if len(sys.argv) != 2:
    print("Usage: python3 ./hits.py <file.mtx>")
    exit()

graph_coo = mmread(sys.argv[1])
print("Loading COO matrix")
print(graph_coo.nnz, " edges")

graph_nx = nx.DiGraph(graph_coo)

print("Creating NetworkX Graph")
print("NetworkX is Directed: ", nx.is_directed(graph_nx))
print("NetworkX Graph has ", graph_nx.number_of_edges(), " edges")

max_iter = 10000
tol = 1e-6
hubs_nx, auths_nx = hits(graph_nx, max_iter, tol, normalized=True)

# Numpy implementation
hrank = np.zeros((graph_coo.shape[0], 1))
arank = np.zeros((graph_coo.shape[0], 1))

hrank += 1/graph_coo.shape[0]
arank += 1/graph_coo.shape[0]

for _ in range(0, max_iter):
    hlast = hrank
    alast = arank
    hrank = np.zeros((graph_coo.shape[0], 1))
    arank = np.zeros((graph_coo.shape[0], 1))

    for edge in range(0, graph_coo.nnz):
        src = int(graph_coo.row[edge])
        dest = int(graph_coo.col[edge])
        arank[dest] += hlast[src]
        hrank[src] += alast[dest]

    # Normalize
    hrank = hrank / np.max(hrank)
    arank = arank / np.max(arank)

    err = np.sum(np.absolute(hrank-hlast))
    if err < tol:
        break

hrank = hrank / np.linalg.norm(hrank, ord=1)
arank = arank / np.linalg.norm(arank, ord=1)

hubs_np = {}
auths_np = {}

for i in range(0, graph_coo.shape[0]):
    hubs_np[i] = hrank[i]
    auths_np[i] = arank[i]

print("Hubs: ")
for key, val in sorted(hubs_nx.items(), key=lambda x: x[1], reverse=True):
    print(key, val, hubs_np[key])
print("Authorities: ")
for key, val in sorted(auths_nx.items(), key=lambda x: x[1], reverse=True):
    print(key, val, auths_np[key])