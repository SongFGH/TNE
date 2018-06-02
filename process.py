import os
from os.path import basename, splitext, join
from tne.tne import TNE
from utils.utils import *
import time


def run(args):

    dataset_path = args.input
    outputs_folder = args.output

    params = {}
    # The method name
    params['method'] = args.method
    # Common parameters
    params['number_of_walks'] = args.n
    params['walk_length'] = args.l
    params['window_size'] = args.w
    params['embedding_size'] = args.d
    params['number_of_topics'] = args.k
    # Parameters for Deepwalk
    params['dw_alpha'] = args.dw_alpha
    # Parameters for Node2vec
    params['n2v_p'] = args.n2v_p
    params['n2v_q'] = args.n2v_q
    # Parameters for LDA
    params['lda_alpha'] = float(50.0 / params['number_of_topics']) if args.lda_alpha == -1.0 else args.lda_alpha
    params['lda_beta'] = args.lda_beta
    params['lda_number_of_iters'] = args.lda_iter_num
    params['emb'] = args.emb

    temp_folder = "./temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    # output folder
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    sub_run(dataset_path, outputs_folder, temp_folder, params)


def sub_run(graph_path, main_outputs_folder, main_temp_folder, params):

    graph_name = splitext(basename(graph_path))[0]
    file_desc = "{}_n{}_l{}_w{}_k{}_{}".format(graph_name,
                                               params['number_of_walks'],
                                               params['walk_length'],
                                               params['window_size'],
                                               params['number_of_topics'],
                                               params['method'])
    # temp folder
    sub_temp_folder = os.path.join(main_temp_folder, file_desc)
    if not os.path.exists(sub_temp_folder):
        os.makedirs(sub_temp_folder)
    temp_folder = sub_temp_folder

    # output folder
    sub_output_folder = os.path.join(main_outputs_folder, file_desc)
    if not os.path.exists(sub_output_folder):
        os.makedirs(sub_output_folder)
    outputs_folder = sub_output_folder

    # Embedding files
    node_embedding_file = join(outputs_folder, "{}_node.embedding".format(file_desc))
    topic_embedding_file = join(outputs_folder, "{}_topic.embedding".format(file_desc))
    # Concatenated embeddings
    concatenated_embedding_file_max = join(outputs_folder, "{}_final_max.embedding".format(file_desc))
    concatenated_embedding_file_avg = join(outputs_folder, "{}_final_avg.embedding".format(file_desc))
    concatenated_embedding_file_min = join(outputs_folder, "{}_final_min.embedding".format(file_desc))
    # Corpus files
    corpus_path_for_node = join(temp_folder, "{}_node_corpus.corpus".format(file_desc))
    corpus_path_for_lda = join(temp_folder, "{}_lda_corpus.corpus".format(file_desc))

    tne = TNE(graph_path, params)
    tne.perform_random_walks(node_corpus_path=corpus_path_for_node)
    tne.save_corpus(corpus_path_for_lda, with_title=True)
    id2node = tne.run_lda(lda_corpus_path=corpus_path_for_lda)
    tne.learn_node_embedding(node_corpus_path=corpus_path_for_node, node_embedding_file=node_embedding_file)
    tne.learn_topic_embedding(node_corpus_path=corpus_path_for_node,
                              topic_embedding_file=topic_embedding_file)

    phi_file = tne.get_file_path(filename='phi')

    if params['emb'] == 'max' or params['emb'] == 'all':
        # Compute the corresponding topics for each node
        initial_time = time.time()
        node2topic_max = find_topics_for_nodes(phi_file, id2node, params['number_of_topics'], type="max")
        # Concatenate the embeddings
        concatenate_embeddings(node_embedding_file=node_embedding_file,
                               topic_embedding_file=topic_embedding_file,
                               node2topic=node2topic_max,
                               output_filename=concatenated_embedding_file_max)
        print("-> The final_max embeddings were generated and saved in {:.2f} secs | {}".
              format((time.time()-initial_time), concatenated_embedding_file_max))

    if params['emb'] == 'avg' or params['emb'] == 'all':
        # Concatenate the embeddings
        initial_time = time.time()
        concatenate_embeddings_avg(node_embedding_file=node_embedding_file,
                                   topic_embedding_file=topic_embedding_file,
                                   phi_file=phi_file,
                                   id2node=id2node,
                                   output_filename=concatenated_embedding_file_avg)
        print("-> The final_avg embeddings were generated and saved in {:.2f} secs | {}".
              format((time.time()-initial_time), concatenated_embedding_file_avg))

    if params['emb'] == 'min' or params['emb'] == 'all':
        # Concatenate the embeddings
        initial_time = time.time()
        node2topic_min = find_topics_for_nodes(phi_file, id2node, params['number_of_topics'], type="min")
        concatenate_embeddings(node_embedding_file=node_embedding_file,
                               topic_embedding_file=topic_embedding_file,
                               node2topic=node2topic_min,
                               output_filename=concatenated_embedding_file_min)
        print("-> The final_min embeddings were generated and saved in {:.2f} secs | {}".
              format((time.time()-initial_time), concatenated_embedding_file_min))

