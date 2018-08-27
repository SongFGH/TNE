import os
from os.path import basename, splitext, join
from tne.tne import TNE
from utils.utils import *
import time

dataset_folder = "../datasets/"
dataset_file = "citeseer.gml"

outputs_folder = "../outputs/"
temp_folder = "../temp/"

# Set all parameters #
params = {}
params['method'] = "node2vec"
# Common parameters
params['number_of_walks'] = 80
params['walk_length'] = 10
params['window_size'] = 10
params['embedding_size'] = 128
params['number_of_topics'] = 80
# Parameters for LDA
params['lda_number_of_iters'] = 1000
params['lda_alpha'] = 50.0 / float(params['number_of_topics'])
params['lda_beta'] = 0.1
# Parameters for Deepwalk
params['dw_alpha'] = 0
# Parameters for Node2vec
params['n2v_p'] = 1.0
params['n2v_q'] = 1.0

params['hs'] = 0
params['negative'] = 5

# Define the file paths
nx_graph_path = dataset_folder + dataset_file

file_desc = "{}_n{}_l{}_w{}_k{}_{}".format(splitext(basename(dataset_file))[0],
                                           params['number_of_walks'],
                                           params['walk_length'],
                                           params['window_size'],
                                           params['number_of_topics'],
                                           params['method'])

# temp folder
sub_temp_folder = os.path.join(temp_folder, file_desc)
if not os.path.exists(sub_temp_folder):
    os.makedirs(sub_temp_folder)
temp_folder = sub_temp_folder

# output folder
sub_output_folder = os.path.join(outputs_folder, file_desc)
if not os.path.exists(sub_output_folder):
    os.makedirs(sub_output_folder)
outputs_folder = sub_output_folder

node_embedding_file = join(outputs_folder, "{}_node.embedding".format(file_desc))
topic_embedding_file = join(outputs_folder, "{}_topic.embedding".format(file_desc))

concatenated_embedding_file_max = join(outputs_folder, "{}_final_max.embedding".format(file_desc))
concatenated_embedding_file_avg = join(outputs_folder, "{}_final_avg.embedding".format(file_desc))
concatenated_embedding_file_min = join(outputs_folder, "{}_final_min.embedding".format(file_desc))

corpus_path_for_node = join(temp_folder, "{}_node_corpus.corpus".format(file_desc))
corpus_path_for_lda = join(temp_folder, "{}_lda_corpus.corpus".format(file_desc))

# If the temp and output folders not exists, create them
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

####
tne = TNE(nx_graph_path, params)
tne.perform_random_walks(node_corpus_path=corpus_path_for_node)
tne.save_corpus(corpus_path_for_lda, with_title=True)
id2node = tne.run_lda(lda_corpus_path=corpus_path_for_lda)
tne.learn_node_embedding(node_corpus_path=corpus_path_for_node,
                         node_embedding_file=node_embedding_file)
tne.learn_topic_embedding(node_corpus_path=corpus_path_for_node,
                          topic_embedding_file=topic_embedding_file)


number_of_nodes = tne.get_number_of_nodes()
phi_file = tne.get_file_path(filename='phi')

# Compute the corresponding topics for each node
initial_time = time.time()
node2topic_max = find_topics_for_nodes(phi_file, id2node, params['number_of_topics'], type="max")
# Concatenate the embeddings
concatenate_embeddings(node_embedding_file=node_embedding_file,
                       topic_embedding_file=topic_embedding_file,
                       node2topic=node2topic_max,
                       output_filename=concatenated_embedding_file_max)
print("-> The final_max embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_max))

# Concatenate the embeddings
initial_time = time.time()
concatenate_embeddings_wmean(node_embedding_file=node_embedding_file,
                             topic_embedding_file=topic_embedding_file,
                             phi_file=phi_file,
                             id2node=id2node,
                             output_filename=concatenated_embedding_file_avg)
print("-> The final_avg embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_avg))

initial_time = time.time()
# Concatenate the embeddings
node2topic_min = find_topics_for_nodes(phi_file, id2node, params['number_of_topics'], type="min")
concatenate_embeddings(node_embedding_file=node_embedding_file,
                       topic_embedding_file=topic_embedding_file,
                       node2topic=node2topic_min,
                       output_filename=concatenated_embedding_file_min)
print("-> The final_min embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_min))