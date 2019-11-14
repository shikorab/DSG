# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../..")


from datasets import models
sys.modules['Data.Visualgenome.models'] = models

import os

import cPickle

from datasets.imdb import imdb
from datasets.datarr import datarr
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from model.config import cfg
import cv2
import json

class visualgenome(datarr):
  def __init__(self, image_set):
    datarr.__init__(self, "VisualGenome", 100, 70, image_set)

  def set_paths(self):
    self._annotations_path = os.path.join(self._devkit_path, "annotations_" + self._image_set + ".json")
    self._data_path = os.path.join(self._devkit_path, "JPEGImages")
    self._im_metadata_path = os.path.join(self._devkit_path, self._image_set + "_image_metadata.json")

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, "VG_100K", index)
    if not os.path.exists(image_path):
        image_path = os.path.join(self._data_path, "VG_100K_2", index)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path
