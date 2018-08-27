# TNE
TNE: A Latent Model for Representation Learning on Networks

#### Installation
Run the following commands to clone [Deepwalk](https://github.com/phanein/deepwalk) and [Node2vec](https://github.com/aditya-grover/node2vec) into the "datasets" folder.

git clone https://github.com/phanein/deepwalk.git ext/deepwalk

git clone https://github.com/aditya-grover/node2vec.git ext/node2vec

Similarly, download [GibbsLda++](http://gibbslda.sourceforge.net/) into the 'dataset' folder.

### How to run
python run.py --input datasets/karate.gml --output ./outputs --method=deepwalk --n 10 --l 5 --k 2 --lda_iter_num 100 --emb=max

