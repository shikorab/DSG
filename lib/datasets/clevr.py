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
sys.modules['Data.Clevr.models'] = models

import os

from datasets.datarr import datarr


class clevr(datarr):
  def __init__(self, image_set):
    datarr.__init__(self, "clevr", 48, 4, image_set)

  def set_paths(self):
    self._annotations_path = os.path.join(self._devkit_path, "annotations_" + self._image_set + ".json")
    self._data_path = os.path.join(self._devkit_path, "CLEVR_v1.0", "images", {"test": "val", "train" : "train"}[self._image_set])
    self._im_metadata_path = os.path.join(self._devkit_path, self._image_set + "_image_metadata.json")

