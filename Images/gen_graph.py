import random
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from networkx import bipartite


def is_planar(G):
    result=True
    bad_minor=[]
    n=len(G.nodes())
    if n>5:
        for subnodes in it.combinations(G.nodes(),6):
            subG=G.subgraph(subnodes)
            if bipartite.is_bipartite(G):# check if the graph G has a subgraph K(3,3)
                X, Y = bipartite.sets(G)
                if len(X)==3:
                    result=False
                    bad_minor=subnodes
    if n>4 and result:
        for subnodes in it.combinations(G.nodes(),5):
            subG=G.subgraph(subnodes)
            if len(subG.edges())==10:# check if the graph G has a subgraph K(5)
                result=False
                bad_minor=subnodes
    return result,bad_minor

p=0.6

for i in range(100):
    while True:
        G = nx.gnp_random_graph(random.choice(range(5,9)), p)
        if is_planar(G)[0] and nx.is_connected(G):
            break

    nx.draw(G, node_color='k', node_size=25)
    plt.savefig("images/graph{}.png".format(i))
    plt.clf()



