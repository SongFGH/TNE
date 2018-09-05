import numpy as np
import networkx as nx
import random


class Graph:
    def __init__(self, nxg, params):
        self.g = nxg
        self.alias_nodes = {}
        self.alias_edges = {}

        self.p = params['p']
        self.q = params['q']
        self.r = params['r']
        self.number_of_walks = params['number_of_walks']
        self.walk_length = params['walk_length']

        self.prev_list = {node: None for node in self.g.nodes()}
        self.next_list = {node: None for node in self.g.nodes()}

    def walk(self, initial_node):

        walk = [initial_node]

        while len(walk) < self.walk_length:

            current_node = random.choice(walk)


            current_nb_list = sorted(nx.neighbors(self.g, current_node))

            if len(current_nb_list) > 0:

                if len(walk) == 1:
                    walk.append(current_nb_list[alias_draw(self.alias_nodes[current_node][0],
                                                           self.alias_nodes[current_node][1])])
                    self.prev_list[walk[-1]] = walk[-2]
                    self.prev_list[walk[-2]] = walk[-1]

                    self.next_list[walk[-1]] = walk[-2]
                    self.next_list[walk[-2]] = walk[-1]
                else:

                    previous_node = self.prev_list[current_node]

                    J, q = self.get_alias_edge(prev=previous_node, current=current_node, next=self.next_list[current_node])

                    next_node = current_nb_list[alias_draw(J, q)]
                    walk.append(next_node)

                    self.prev_list[next_node] = current_node
                    self.next_list[current_node] = next_node
            else:
                break

        return walk

    def perform_walks(self):
        walks = []

        node_list = list(self.g.nodes())

        for walk_iter in range(self.number_of_walks):
            print str(walk_iter+1), '/', str(self.number_of_walks)
            random.shuffle(node_list)
            for node in node_list:
                walks.append(self.walk(initial_node=node))

        return walks

    def get_alias_edge(self, prev, current, next=None):
        """
        Get the alias edge setup lists for a given edge.
        """

        unnormalized_probs = []

        for current_nbr in sorted(self.g.neighbors(current)):
            if current_nbr == prev or current_nbr == next:
                unnormalized_probs.append(self.g[current][current_nbr]['weight'] / self.p)
            elif self.g.has_edge(current_nbr, prev) or (next is not None and self.g.has_edge(current_nbr, current)):
                unnormalized_probs.append(self.g[current][current_nbr]['weight'] / self.r)
            else:
                unnormalized_probs.append(self.g[current][current_nbr]['weight'] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """

        for node in self.g.nodes():
            unnormalized_probs = [self.g[node][nb]['weight'] for nb in sorted(self.g.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = alias_setup(normalized_probs)

        for edge in self.g.edges():
            self.alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            self.alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        return


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):

    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]