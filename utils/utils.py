import numpy as np
from collections import OrderedDict
from gensim.utils import smart_open


def find_topics_for_nodes(phi_file, id2node, number_of_topics, type):

    number_of_nodes = len(id2node)

    # Phi is the node-topic distribution
    phi = np.zeros(shape=(number_of_topics, number_of_nodes), dtype=np.float)

    i = 0
    with smart_open(phi_file, 'r') as f:
        for vals in f.readlines():
            phi[i, :] = [float(v) for v in vals.strip().split()]
            i += 1

    if type == "max":
        arginx = np.argmax(phi, axis=0)
    if type == "min":
        arginx = np.argmin(phi, axis=0)

    node2topic = {}
    for i in range(arginx.shape[0]):
        node2topic.update({id2node[i]: arginx[i]})

    return node2topic


def generate_id2node(wordmap_file):
    id2node = {}
    with smart_open(wordmap_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            id, node = int(tokens[1]), tokens[0]
            id2node.update({id: node})

    return id2node


def convert_node2topic(tassign_file):
    with smart_open(tassign_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            yield [token.split(':')[1] for token in tokens]


def concatenate_embeddings(node_embedding_file, topic_embedding_file, node2topic, output_filename):

    # Read the node embeddings
    node_embeddings = OrderedDict()
    with smart_open(node_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            node_embeddings.update({tokens[0]: [val for val in tokens[1:]]})

    # Read the topic embeddings
    topic_embeddings = {}
    topic_num = 0
    with smart_open(topic_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embeddings.update({tokens[0]: [val for val in tokens[1:]]})
            topic_num += 1

    # Concatenate the embeddings
    concatenated_embeddings = {}
    for node in node_embeddings:
        concatenated_embeddings.update({node: node_embeddings[node] + topic_embeddings[str(node2topic[node])]})

    number_of_nodes = len(concatenated_embeddings.keys())
    concatenated_embedding_length = len(concatenated_embeddings.values()[0])
    with smart_open(output_filename, 'w') as f:
        f.write(u"{} {}\n".format(number_of_nodes, concatenated_embedding_length))
        for node in node_embeddings:
            f.write(u"{} {}\n".format(node, " ".join(concatenated_embeddings[node])))


def concatenate_embeddings_wmean(node_embedding_file, topic_embedding_file, phi_file, id2node, output_filename):

    # Read the node embeddings
    node_embeddings = OrderedDict()
    with smart_open(node_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            node_embeddings.update({tokens[0]: [val for val in tokens[1:]]})

    # Read the topic embeddings
    topic_embeddings = {}
    topic_num = 0
    with smart_open(topic_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embeddings.update({tokens[0]: [float(val) for val in tokens[1:]]})
            topic_num += 1

    # Phi is the topic-node distribution
    phi = np.zeros(shape=(topic_num, len(node_embeddings.keys())), dtype=np.float)
    t = 0
    with smart_open(phi_file, 'r') as f:
        for vals in f.readlines():
            phi[t, :] = [float(v) for v in vals.strip().split()]
            t += 1

    # Concatenate the embeddings
    concatenated_embeddings = {}
    number_of_nodes = len(node_embeddings.keys())
    d = len(topic_embeddings['0'])
    for idx in range(number_of_nodes):
        wmean_topic_vec = np.zeros(shape=d, dtype=np.float)
        for t in range(topic_num):
            wmean_topic_vec += np.multiply(topic_embeddings[str(t)], phi[t, idx])

        concatenated_embeddings.update({id2node[idx]: node_embeddings[id2node[idx]] + wmean_topic_vec.tolist()})

    concatenated_embedding_length = len(concatenated_embeddings.values()[0])
    with smart_open(output_filename, 'w') as f:
        f.write("{} {}\n".format(number_of_nodes, concatenated_embedding_length))
        for node in node_embeddings:
            f.write("{} {}\n".format(node, " ".join(str(v) for v in concatenated_embeddings[node])))


def getCommProb(tassing_file_path, lda_num_of_communities):
    # it is assumed that probability of a walk p(w) or probability of a document p(d) is uniform.

    community2counts = np.asarray([0.0 for _ in range(lda_num_of_communities)])

    with smart_open(tassing_file_path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            for token in tokens:
                community_label = token.split(':')[1]
                community2counts[int(community_label)] += 1.0

    return community2counts / np.sum(community2counts)


def getPhi(phi_file_path):
    phi = []  # num_of_communities X num_of_nodes

    with smart_open(phi_file_path, 'r') as f:
        for values in f.readlines():
            phi.append([float(v) for v in values.strip().split()])

    return np.asarray(phi)


def prob_of_vertex_given_community(tassing_file_path, phi_file_path, lda_num_of_communities):

    community_prob = getCommProb(tassing_file_path, lda_num_of_communities)
    phi = getPhi(phi_file_path)

    unnormalized = np.multiply(phi.T, community_prob)
    prob = np.divide(unnormalized, np.sum(unnormalized, 0))
    return prob  # num_of_nodes X num_of_communities


def concatenate_embeddings_wmean2(node_embedding_file, topic_embedding_file, tassing_file, phi_file,
                                  lda_num_of_communities, id2node, output_filename):

    # Read the node embeddings
    node_embeddings = OrderedDict()
    with smart_open(node_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            node_embeddings.update({tokens[0]: [val for val in tokens[1:]]})

    # Read the topic embeddings
    topic_embeddings = {}
    with smart_open(topic_embedding_file, 'r') as f:
        f.readline()  # Skip the first line
        for line in f:
            tokens = line.strip().split()
            # word = int(tokens[0])
            topic_embeddings.update({tokens[0]: [float(val) for val in tokens[1:]]})

    # Get the probability of vertex given community
    prob = prob_of_vertex_given_community(tassing_file_path=tassing_file, phi_file_path=phi_file,
                                          lda_num_of_communities=lda_num_of_communities)

    # Concatenate the embeddings
    concatenated_embeddings = {}
    number_of_nodes = len(node_embeddings.keys())
    topic_emb_dim = len(topic_embeddings['0'])
    for idx in range(number_of_nodes):
        wmean_topic_vec = np.zeros(shape=topic_emb_dim, dtype=np.float)
        for t in range(lda_num_of_communities):
            wmean_topic_vec += np.multiply(topic_embeddings[str(t)], prob[idx, t])

        concatenated_embeddings.update({id2node[idx]: node_embeddings[id2node[idx]] + wmean_topic_vec.tolist()})

    # Write the concatenated embeddings to a file
    concatenated_embedding_length = len(concatenated_embeddings.values()[0])
    with smart_open(output_filename, 'w') as f:
        f.write("{} {}\n".format(number_of_nodes, concatenated_embedding_length))
        for node in node_embeddings:
            f.write("{} {}\n".format(node, " ".join(str(v) for v in concatenated_embeddings[node])))