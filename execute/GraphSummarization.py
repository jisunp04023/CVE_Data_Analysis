import matplotlib.pyplot as plt
import pandas as pd
from time import time
import networkx as nx
from sklearn.manifold import TSNE

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr

from gem.embedding.hope     import HOPE

from sklearn.datasets import make_blobs
import numpy as np
import math
import scipy as sp
import seaborn as sns

from sklearn.cluster import KMeans

# File that contains the edges. Format: source target
# Optionally, you can add weights as third column: source target weight
edge_f = './data/CVE_Edge.csv'
# Specify whether the edges are directed
isDirected = False

# Load graph
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
G = G.to_directed()

models = []
models.append(HOPE(d=4, beta=0.01))

def embedding2D(node_pos, node_colors=None, di_graph=None, labels=None):
    node_num, embedding_dimension = node_pos.shape
    if(embedding_dimension > 2):
        #print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in range(node_num):
            pos[i] = node_pos[i, :]
        return pos

for embedding in models:
    print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
    t1 = time()
    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    # Evaluate on graph reconstruction
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
    #---------------------------------------------------------------------------------
    print(("\tMAP: {} \t preccision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
    #---------------------------------------------------------------------------------
    # Visualize
    viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
    #print(embedding.get_embedding()[:, :2])
    plt.savefig("./results/Embedding.png")
    plt.show()
    plt.clf()

pos = embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
Embedding = pd.DataFrame(pos)
Embedding = Embedding.T
Embedding.columns = ['x', 'y']

kmeans = KMeans(n_clusters=6) # k 설정
kmeans.fit(Embedding)

result_by_sklearn = Embedding.copy()
result_by_sklearn['Cluster'] = kmeans.labels_

#%matplotlib inline
splot = sns.scatterplot(x="x", y="y", hue="Cluster", data=result_by_sklearn, palette="Set2")
figure = splot.get_figure()
figure.savefig('./results/Clustering.png', orientation = 'landscape')

data = pd.read_csv("./data/Label.csv")
Label = pd.DataFrame(data)

ClusterTable = pd.concat([Label, result_by_sklearn], axis = 1)

ClusterTable.to_csv("./results/ClusterTable.csv", index = True)