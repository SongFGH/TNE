from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import process

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
    parser.add_argument('--dw_alpha', type=int, default=0,
                        help='The parameter for Deepwalk')
    parser.add_argument('--n2v_p', type=int, default=0,
                        help='The parameter for node2vec')
    parser.add_argument('--n2v_q', type=int, default=0,
                        help='The parameter for node2vec')
    parser.add_argument('--sg', type=int, default=1,
                        help='The training algorithm, 1 for sg, otherwise CBOW')
    parser.add_argument('--negative', type=int, default=0,
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

    process.run(args)