# Sentence Representations

## Data Sources

Downliad and unzip from https://drive.google.com/file/d/1eQBkC3n4z5BDpJ7Z19fmfyhI_t8VCgFc/view?usp=sharing

## Random

From the random directory, simply run build_random_space.py -s {dataset}.

## GEM
From the gem directory, simply run build_gem_space.py -s {dataset}.

## GloVe
From the glove directory, simply run build_glove_space.py -s {dataset}.

## GloVe-DCT
From the dct directory, simply run build_dct_space.py -s {dataset}. This will build the spaces for $k in {0-6}$.


## SentBERT
Install the sentence-transformer module as outlined in: https://github.com/UKPLab/sentence-transformers
From the sentbert directory, run build_sentbert_space.py -s {dataset}.

## LASER
Install the laserembeddings module following the instructions here: https://github.com/yannvgn/laserembeddings. Download the necessary models using python -m laserembeddings download-models. From
the laser directory, run build_laser_space.py -s {dataset}.

## QuickThought
Clone the S2V repository and follow their instructions for downloading the pre-trained models: https://github.com/lajanugen/S2V.
Within the src directory, add the four python files from this repository.
Finally, run build_quickthought_space.py -s {dataset}.

## InferSent
Clone the infersent repository https://github.com/facebookresearch/InferSenand follow their instructions for fetching the pre-trained models (GloVe and FastText).
Then, run build_quickthought_space.py -s {dataset} -v {infersent_version} -p {path_to_vectors}.

## SkipThought
Clone the skipthoughts repository: https://github.com/ryankiros/skip-thoughts. Create a directory called models and follow their instructions for downloading the pre-trained data assets.
Copy the python file in this repository into the skip-thoughts directory and run build_skipthought_space.py -s {dataset}.