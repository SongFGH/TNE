from os.path import basename, splitext
from tne.tne import TNE
from utils.utils import *
import time

dataset_folder = "../datasets/"
outputs_folder = "../outputs/"
temp_folder = "../temp/"

dataset_file = "citeseer.gml"

method = "deepwalk"
number_of_topics = 65
number_of_iters = 1000
params = {}
params['number_of_walks'] = 40
params['walk_length'] = 10

params['window_size'] = 10
params['embedding_size'] = 128
params['alpha'] = 0
params['p'] = 1.0
params['q'] = 1.0
alpha = 50.0/float(number_of_topics)
beta = 0.1


node_embedding_file = outputs_folder + \
                      "{}_n{}_l{}_w{}_k{}_{}_node.embedding".format(splitext(basename(dataset_file))[0],
                                                                    params['number_of_walks'],
                                                                    params['walk_length'],
                                                                    params['window_size'],
                                                                    number_of_topics,
                                                                    method)

topic_embedding_file = outputs_folder + \
                      "{}_n{}_l{}_w{}_k{}_{}_topic.embedding".format(splitext(basename(dataset_file))[0],
                                                                     params['number_of_walks'],
                                                                     params['walk_length'],
                                                                     params['window_size'],
                                                                     number_of_topics,
                                                                     method)

corpus_path_for_lda = temp_folder + \
                      "{}_n{}_l{}_w{}_k{}_{}.corpus".format(splitext(basename(dataset_file))[0],
                                                            params['number_of_walks'],
                                                            params['walk_length'],
                                                            params['window_size'],
                                                            number_of_topics,
                                                            method)

concatenated_embedding_file_max = outputs_folder + \
                      "{}_n{}_l{}_w{}_k{}_{}_final_max.embedding".format(splitext(basename(dataset_file))[0],
                                                                     params['number_of_walks'],
                                                                     params['walk_length'],
                                                                     params['window_size'],
                                                                     number_of_topics,
                                                                     method)

concatenated_embedding_file_avg = outputs_folder + \
                      "{}_n{}_l{}_w{}_k{}_{}_final_avg.embedding".format(splitext(basename(dataset_file))[0],
                                                                     params['number_of_walks'],
                                                                     params['walk_length'],
                                                                     params['window_size'],
                                                                     number_of_topics,
                                                                     method)

concatenated_embedding_file_min = outputs_folder + \
                      "{}_n{}_l{}_w{}_k{}_{}_final_min.embedding".format(splitext(basename(dataset_file))[0],
                                                                     params['number_of_walks'],
                                                                     params['walk_length'],
                                                                     params['window_size'],
                                                                     number_of_topics,
                                                                     method)


graph_path = dataset_folder + dataset_file
tne = TNE(graph_path)
tne.perform_random_walks(method="deepwalk", params=params)
tne.save_corpus(corpus_path_for_lda, with_title=True)
id2node = tne.run_lda(alpha=alpha, beta=beta, number_of_iters=number_of_iters,
                      number_of_topics=number_of_topics, lda_corpus_path=corpus_path_for_lda)
tne.extract_node_embedding(node_embedding_file)
tne.extract_topic_embedding(number_of_topics=number_of_topics,
                            topic_embedding_file=topic_embedding_file)


number_of_nodes = tne.number_of_nodes
phi_file = tne.lda_phi_file

# Compute the corresponding topics for each node
initial_time = time.time()
node2topic_max = find_max_topic_for_nodes(phi_file, id2node, number_of_nodes, number_of_topics)
# Concatenate the embeddings
concatenate_embeddings_max(node_embedding_file=node_embedding_file,
                           topic_embedding_file=topic_embedding_file,
                           node2topic=node2topic_max,
                           output_filename=concatenated_embedding_file_max)
print("-> The final_max embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_max))

# Concatenate the embeddings
initial_time = time.time()
concatenate_embeddings_avg(node_embedding_file=node_embedding_file,
                           topic_embedding_file=topic_embedding_file,
                           phi_file=phi_file,
                           id2node=id2node,
                           output_filename=concatenated_embedding_file_avg)
print("-> The final_avg embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_avg))

initial_time = time.time()
node2topic_min = find_min_topic_for_nodes(phi_file, id2node, number_of_nodes, number_of_topics)
# Concatenate the embeddings

concatenate_embeddings_min(node_embedding_file=node_embedding_file,
                           topic_embedding_file=topic_embedding_file,
                           node2topic=node2topic_min,
                           output_filename=concatenated_embedding_file_min)
print("-> The final_min embeddings were generated and saved in {:.2f} secs | {}".format((time.time()-initial_time), concatenated_embedding_file_min))