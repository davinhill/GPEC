import numpy as np
import warnings
import tqdm
try:
    import networkx as nx
except:
    warnings.warn('Networkx is not installed')
try:
    from sklearn.neighbors import NearestNeighbors
except:
    warnings.warn('Sklearn is not installed')
try:
    import matplotlib.pyplot as plt
except:
    warnings.warn('Matplotlib is not installed')
#from transformers import PreTrainedTokenizerBase

"""
adapted from:

Fast Approximate Geodesics for Deep Generative Models
Nutan Chen, Francesco Ferroni, Alexej Klushyn, Alexandros Paraschos, Justin Bayer, Patrick van der Smagt

and corresponding code: https://github.com/redst4r/riemannian_latent_space

"""

class RiemannianMetric(object):
    def __init__(self):
        pass

    def riemannian_distance_along_line(self, z1, z2, n_steps):
        """
        calculates the riemannian distance between two near points in latent space on a straight line
        the formula is L(z1, z2) = \int_0^1 dt \sqrt(\dot \gamma^T J^T J \dot gamma)
        since gamma is a straight line \gamma(t) = t z_1 + (1-t) z_2, we get
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T J^T J [z1-z2])
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T G [z1-z2])

        z1: starting point
        z2: end point
        n_steps: number of discretization steps of the integral
        """

        # discretize the integral aling the line
        t = np.linspace(0, 1, n_steps)
        dt = t[1] - t[0]
        the_line = np.concatenate([_ * z1 + (1 - _) * z2 for _ in t])

        #G_eval = self.session.run(self.G, feed_dict={self.z: the_line})
        #G_eval = np.zeros((5,5))

        # eval the integral at discrete point
        L_discrete = np.sqrt((z1-z2) @ (z1-z2).T)
        L_discrete = L_discrete.flatten()

        L = np.sum(dt * L_discrete)

        return L


class RiemannianTree(object):
    """docstring for RiemannianTree"""

    def __init__(self, riemann_metric = None):
        super(RiemannianTree, self).__init__()
        self.riemann_metric = riemann_metric  # decoder input (tf_variable)
        


    def create_riemannian_graph(self, z, n_steps=100, n_neighbors=10):

        n_data = len(z)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(z)

        G = nx.Graph()

        # Nodes
        for i in range(n_data):
            n_attr = {f'z{k}': float(z[i, k]) for k in range(z.shape[1])}
            G.add_node(i, **n_attr)

        # edges
        for i in tqdm.trange(n_data, desc='define_rTree'):
            distances, indices = knn.kneighbors(z[i:i+1])
            # first dim is for samples (z), but we only have one
            distances = distances[0]
            indices = indices[0]

            for ix, dist in zip(indices, distances):
                # calculate the riemannian distance of z[i] and its nn

                # save some computation if we alrdy calculated the other direction
                if (i, ix) in G.edges or (ix, i) in G.edges or i == ix:
                    continue

                L_euclidean = dist

                C = 1e10  # numerical stability
                # note nn-distances are NOT symmetric
                edge_attr = {'weight_euclidean': float(1/max(L_euclidean, 1/C)),
                            'distance_euclidean': float(L_euclidean)}
                if self.riemann_metric is not None:
                    L_riemann = self.riemann_metric.riemannian_distance_along_line(z[i:i+1], z[ix:ix+1], n_steps=n_steps)
                    edge_attr['weight'] = float(1/L_riemann)
                    edge_attr['distance_riemann'] = float(L_riemann),
                G.add_edge(i, ix, **edge_attr)
        return G
