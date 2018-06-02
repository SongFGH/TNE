# TNE
TNE: A Latent Model for Representation Learning on Networks

#### Installation
git clone https://github.com/phanein/deepwalk.git ext/deepwalk
git clone https://github.com/aditya-grover/node2vec.git ext/node2vec

### How to run
python run.py --input datasets/citeseer.gml --output ./outputs --method=deepwalk --n 10 --l 10 --k 10 --lda_iter_num 50


