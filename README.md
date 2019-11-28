# Differentiable Scene Graphs
####  Moshiko Raboh*, [Roei Herzig*](https://roeiherz.github.io/), [Gal Chechik](https://chechiklab.biu.ac.il/~gal/), [Jonathan Berant](http://www.cs.tau.ac.il/~joberant/), [Amir Globerson](http://www.cs.tau.ac.il/~gamir/)<img src="sg_example_final.png" width="750">

## Introduction
TBD

## Model implementation
TBD
"Differentiable Scene Graphs" implemented on top of "https://github.com/endernewton/tf-faster-rcnn".

## Dependencies
TBD

## Setup

# Compile Cyton
```
cd lib
make clean
make
cd ..
```

# Download image-net weights
```
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```
# download vg
```
cd data
wget https://cs.stanford.edu/people/ranjaykrishna/referringrelationships/visualgenome.zip
unzip visualgenome.zip
rm visualgenome.zip
cd VisualGenome
mkdir JPEGImages
cd JPEGImages
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip
rm images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip
rm images2.zip
cd ../../
```

## Usage
Train a model:
`./experiments/scripts/train.sh <gpu-id> visualgenome res101 <experiment name>`

Test a model:
`./experiments/scripts/train.sh <gpu-id> visualgenome res101 <experiment name>_iter_0`

Test a pre trained model:
`./experiments/scripts/train.sh <gpu-id> visualgenome res101 dsg_pretrained`

## About this repository
TBD

## Cite
Please cite our paper if you use this code in your own work:
```
@article{DBLP:journals/corr/abs-1902-10200,
  author    = {Moshiko Raboh and
               Roei Herzig and
               Gal Chechik and
               Jonathan Berant and
               Amir Globerson},
  title     = {Learning Latent Scene-Graph Representations for Referring Relationships},
  journal   = {CoRR},
  volume    = {abs/1902.10200},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.10200},
  archivePrefix = {arXiv},
  eprint    = {1902.10200},
  timestamp = {Tue, 21 May 2019 18:03:39 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1902-10200},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```