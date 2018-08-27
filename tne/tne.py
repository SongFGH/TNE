import os
import sys
import time
import random
import networkx as nx
from utils.utils import *
from ext.gensim_wrapper.models.word2vec import Word2VecWrapper, CombineSentences, LineSentence
from gensim.utils import smart_open

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/deepwalk/deepwalk")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/node2vec/src")))
lda_exe_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ext/gibbslda/lda"))


try:
    import graph as deepwalk
    import node2vec
    if not os.path.exists(lda_exe_path):
        raise ImportError
except ImportError:
    raise ImportError("An error occurred during loading the external libraries!")


class WalkIterator:
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for walk in self.corpus:
            yield walk


class TNE:
    def __init__(self, graph_path=None, params={}):
        self.graph = None
        self.graph_name = ""
        self.number_of_nodes = 0
        self.corpus = []
        self.params = params
        self.model = None

        self.temp_folder = "../temp/"

        self.lda_corpus_dir = ""
        self.lda_wordmapfile = ""
        self.lda_tassignfile = ""
        self.lda_topic_corpus_path = ""
        self.lda_phi_file = ""
        self.lda_theta_file = ""

        if graph_path is not None:
            self.read_graph(graph_path)

        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def read_graph(self, filename, filetype=".gml"):
        dataset_name = os.path.splitext(os.path.basename(filename))[0]

        if filetype == ".gml":
            g = nx.read_gml(filename)
            print("Dataset: {}".format(dataset_name))
            print("The number of nodes: {}".format(g.number_of_nodes()))
            print("The number of edges: {}".format(g.number_of_edges()))

            self.number_of_nodes = g.number_of_nodes()
            self.graph = g
            self.graph_name = dataset_name
        else:
            raise ValueError("Invalid file type!")

    def set_graph(self, graph, graph_name="unknown"):

        self.graph = graph
        self.number_of_nodes = graph.number_of_nodes()
        self.graph_name = graph_name

        print("Graph name: {}".format(self.graph_name))
        print("The number of nodes: {}".format(self.graph.number_of_nodes()))
        print("The number of edges: {}".format(self.graph.number_of_edges()))

    def set_params(self, params):
        self.params = params

    def perform_random_walks(self, node_corpus_path):

        initial_time = time.time()
        # Generate a corpus

        if self.params['method'] == "deepwalk":
            if not ('number_of_walks' and 'walk_length' and 'dw_alpha') in self.params.keys():
                raise ValueError("A parameter is missing!")

            # Temporarily generate the edge list
            with open(self.temp_folder + "graph_deepwalk.edgelist", 'w') as f:
                for line in nx.generate_edgelist(self.graph, data=False):
                    f.write("{}\n".format(line))

            dwg = deepwalk.load_edgelist(self.temp_folder + "graph_deepwalk.edgelist", undirected=True)
            self.corpus = deepwalk.build_deepwalk_corpus(G=dwg, num_paths=self.params['number_of_walks'],
                                                         path_length=self.params['walk_length'],
                                                         alpha=self.params['dw_alpha'],
                                                         rand=random.Random(0))

        elif self.params['method'] == "node2vec":

            if not ('number_of_walks' and 'walk_length' and 'n2v_p' and 'n2v_q') in self.params.keys():
                raise ValueError("A missing parameter exists!")

            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['weight'] = 1
            G = node2vec.Graph(nx_G=self.graph, p=self.params['n2v_p'], q=self.params['n2v_q'], is_directed=False)
            G.preprocess_transition_probs()
            self.corpus = G.simulate_walks(num_walks=self.params['number_of_walks'],
                                           walk_length=self.params['walk_length'])

        else:
            raise ValueError("Invalid method name!")

        self.save_corpus(node_corpus_path, with_title=False)

        print("The corpus was generated in {:.2f} secs.".format(time.time() - initial_time))

    def save_corpus(self, corpus_file, with_title=False, corpus=None):

        # Save the corpus
        with open(corpus_file, "w") as f:

            if with_title is True:
                f.write(u"{}\n".format(self.number_of_nodes * self.params['number_of_walks']))

            if corpus is None:
                for walk in self.corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))
            else:
                for walk in corpus:
                    f.write(u"{}\n".format(u" ".join(v for v in walk)))

    def learn_node_embedding(self, node_corpus_path, node_embedding_file, workers=3):

        initial_time = time.time()

        if 'negative' not in self.params:
            self.params['hs'] = 1
            self.params['negative'] = 0
        else:
            if self.params['negative'] > 0:
                self.params['hs'] = 0
            else:
                self.params['hs'] = 1


        # Extract the node embeddings
        self.model = Word2VecWrapper(sentences=self.corpus,
                                     size=self.params["embedding_size"],
                                     window=self.params["window_size"],
                                     sg=1, hs=self.params['hs'], negative=self.params['negative'],
                                     workers=workers,
                                     min_count=0)

        # Save the node embeddings
        self.model.wv.save_word2vec_format(fname=node_embedding_file)
        print("The node embeddings were generated and saved in {:.2f} secs.".format(time.time() - initial_time))

    def run_lda(self, lda_corpus_path):

        if not ('lda_alpha' and 'lda_beta' and 'lda_number_of_iters' and 'number_of_topics') in self.params.keys():
            raise ValueError("Missing paramater for LDA!")

        self.lda_corpus_dir = os.path.dirname(os.path.join(lda_corpus_path))
        self.lda_wordmapfile = os.path.join(self.lda_corpus_dir, "wordmap.txt")
        self.lda_tassignfile = os.path.join(self.lda_corpus_dir, "model-final.tassign")
        self.lda_topic_corpus_path = os.path.join(self.lda_corpus_dir, "lda_topic.file")
        self.lda_phi_file = os.path.join(self.lda_corpus_dir, "model-final.phi")
        self.lda_theta_file = os.path.join(self.lda_corpus_dir, "model-final.theta")

        initial_time = time.time()
        # Run GibbsLDA++
        cmd = "{} -est ".format(lda_exe_path)
        cmd += "-alpha {} ".format(self.params['lda_alpha'])
        cmd += "-beta {} ".format(self.params['lda_beta'])
        cmd += "-ntopics {} ".format(self.params['number_of_topics'])
        cmd += "-niters {} ".format(self.params['lda_number_of_iters'])
        cmd += "-savestep {} ".format(self.params['lda_number_of_iters']+1)
        cmd += "-dfile {} ".format(lda_corpus_path)
        os.system(cmd)

        # Generate the id2node dictionary
        id2node = generate_id2node(self.lda_wordmapfile)
        print("-> The LDA algorithm run in {:.2f} secs".format(time.time() - initial_time))

        assert len(id2node) == self.number_of_nodes, "LDA could not run well!"

        return id2node

    def generate_topic_corpus(self):
        topic_corpus = []
        with smart_open(self.lda_tassignfile, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                topic_corpus.append([token.split(':')[1] for token in tokens])

        return topic_corpus

    def learn_topic_embedding(self, node_corpus_path, topic_embedding_file):

        if 'number_of_topics' not in self.params.keys():
            raise ValueError("The number of topics was not given!")

        # Define the paths for the files generated by GibbsLDA++
        initial_time = time.time()
        # Convert node corpus to the corresponding topic corpus
        topic_corpus = self.generate_topic_corpus()
        self.lda_topic_corpus_path = os.path.join(self.temp_folder, "topic.corpus")
        self.save_corpus(corpus_file=self.lda_topic_corpus_path,  with_title=False, corpus=topic_corpus)

        # Construct the tuples (word, topic) with each word in the corpus and its corresponding topic assignment
        combined_sentences = CombineSentences(node_corpus_path, self.lda_topic_corpus_path)
        # Extract the topic embeddings
        self.model.train_topic(self.params['number_of_topics'], combined_sentences)
        # Save the topic embeddings
        self.model.wv.save_word2vec_topic_format(fname=topic_embedding_file)
        print("The topic embeddings were generated and saved in {:.2f} secs.".format(time.time() - initial_time))

    def get_file_path(self, filename):

        if filename == "phi":
            return os.path.realpath(self.lda_phi_file)

        if filename == "theta":
            return os.path.realpath(self.lda_theta_file)

    def get_nxgraph(self):
        return self.graph

    def get_topic_corpus_path(self):
        return self.lda_topic_corpus_path
