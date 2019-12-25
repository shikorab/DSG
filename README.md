# Differentiable Scene Graphs
####  Moshiko Raboh*, [Roei Herzig*](https://roeiherz.github.io/), [Gal Chechik](https://chechiklab.biu.ac.il/~gal/), [Jonathan Berant](http://www.cs.tau.ac.il/~joberant/), [Amir Globerson](http://www.cs.tau.ac.il/~gamir/)<img src="figure1_iccv.jpg" width="750">

## Introduction
We propose an intermediate “graph-like” representation (DSGs) that can be learned in an end-to-end manner from the supervision for a downstream visual reasoning task, which achieves a new state-of-the-art results on Referring Relationships task.

## Model implementation
TBD
"Differentiable Scene Graphs" implemented on top of "https://github.com/endernewton/tf-faster-rcnn".

## Dependencies
TBD

## Setup

### Compile Cyton
```
cd lib
make clean
make
cd ..
```

### Download image-net weights
```
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```
### Download Visual Genome Dataset
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
@InProceedings{raboh2020dsg,
  title = {Differentiable Scene Graphs},
  author = {Moshiko Raboh and
            Roei Herzig and
            Gal Chechik and
            Jonathan Berant and
            Amir Globerson},
  booktitle = {Winter Conference on Applications of Computer Vision},
  year = {2020}
}
```
