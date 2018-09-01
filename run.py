import os
from os.path import basename, splitext, join
from tne.tne import TNE
from utils.utils import *
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def process(args, temp_folder="./temp"):

    dataset_path = args.input
    outputs_folder = args.output

    # Set the parameters
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
    # Parameters for TNE, common
    params['hs'] = args.hs
    params['negative'] = args.negative
    # Parameters for LDA
    params['lda_alpha'] = float(50.0 / params['number_of_topics']) if args.lda_alpha == -1.0 else args.lda_alpha
    params['lda_beta'] = args.lda_beta
    params['lda_number_of_iters'] = args.lda_iter_num
    params['emb'] = args.emb

    graph_name = splitext(basename(dataset_path))[0]
    file_desc = "{}_n{}_l{}_w{}_k{}_{}".format(graph_name,
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

    tne = TNE(dataset_path, params)
    tne.perform_random_walks(node_corpus_path=corpus_path_for_node)
    tne.save_corpus(corpus_path_for_lda, with_title=True)
    id2node = tne.run_lda(lda_corpus_path=corpus_path_for_lda)
    tne.learn_node_embedding(node_corpus_path=corpus_path_for_node,
                             node_embedding_file=node_embedding_file)
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
        concatenate_embeddings_wmean(node_embedding_file=node_embedding_file,
                                     topic_embedding_file=topic_embedding_file,
                                     phi_file=phi_file,
                                     id2node=id2node,
                                     output_filename=concatenated_embedding_file_avg)
        print("-> The final_avg embeddings were generated and saved in {:.2f} secs | {}".
              format((time.time()-initial_time), concatenated_embedding_file_avg))

        # Concatenate the embeddings2
        initial_time = time.time()
        concatenated_embedding_file_avg2 = join(outputs_folder, "{}_final_avg2.embedding".format(file_desc))
        concatenate_embeddings_wmean2(node_embedding_file=node_embedding_file,
                                      topic_embedding_file=topic_embedding_file,
                                      tassing_file=tne.lda_tassignfile,
                                      phi_file=phi_file,
                                      lda_num_of_communities=params['number_of_topics'],
                                      id2node=id2node,
                                      output_filename=concatenated_embedding_file_avg2)
        print("-> The final_avg2 embeddings were generated and saved in {:.2f} secs | {}".
              format((time.time() - initial_time), concatenated_embedding_file_avg2))

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


def parse_arguments():
    parser = ArgumentParser(description="TNE: A Latent Model for Representation Learning on Networks",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True,
                        help='The graph path')
    parser.add_argument('--output', type=str, required=True,
                        help='The path of the output folder')
    parser.add_argument('--method', choices=['deepwalk', 'node2vec'], required=True,
                        help='The name of the method used for performing random walks')
    parser.add_argument('--n', type=int, required=True,
                        help='The number of walks')
    parser.add_argument('--l', type=int, required=True,
                        help='The length of each walk')
    parser.add_argument('--w', type=int, default=10,
                        help='The window size')
    parser.add_argument('--d', type=int, default=128,
                        help='The size of the embedding vector')
    parser.add_argument('--k', type=int, required=True,
                        help='The number of clusters')
    parser.add_argument('--dw_alpha', type=float, default=0.0,
                        help='The parameter for Deepwalk')
    parser.add_argument('--n2v_p', type=float, default=1.0,
                        help='The parameter for node2vec')
    parser.add_argument('--n2v_q', type=float, default=1.0,
                        help='The parameter for node2vec')
    parser.add_argument('--hs', type=int, default=0,
                        help='1 for the hierachical softmax, 1, otherwise 0 for negative sampling')
    parser.add_argument('--negative', type=int, default=5,
                        help='It specifies how many noise words are used')
    parser.add_argument('--lda_alpha', type=float, default=-1.0,
                        help='A hyperparameter of LDA')
    parser.add_argument('--lda_beta', type=float, default=0.1,
                        help='A hyperparameter of LDA')
    parser.add_argument('--lda_iter_num', type=int, default=2000,
                        help='The number of iterations for GibbsLDA++')
    parser.add_argument('--emb', type=str, default='all',
                        help='Specifies the output embeddings')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    process(args)

