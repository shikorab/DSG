# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.resnet_v1 import resnetv1
from random import shuffle

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--model', dest='model',
                      help='model to load',
                      type=str)
  parser.add_argument('--imdbtest', dest='imdbtest_name',
                      help='dataset to test on',
                      default='visualgenome_test', type=str)
  parser.add_argument('--net', dest='net',
                      help='res101',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb, imdb

  roidb, imdb = get_roidb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':
  VAL_PERCENT = 0.1

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)
  # test set
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb = combined_roidb(args.imdbtest_name)
  print('{:d} test roidb entries'.format(len(roidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  cfg.TRAIN.SNAPSHOT_PREFIX = ""
  cfg.TRAIN.SNAPSHOT_LOAD_PREFIX = "" 


  net = resnetv1(imdb.nof_ent_classes, imdb.nof_rel_classes, num_layers=101)
 

  train_net(net, imdb, [], roidb, "", "",
            pretrained_model=args.model,
            max_iters=1,
            just_test=True)
