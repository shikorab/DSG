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
sys.modules['Data.Datarr.models'] = models

import os

import cPickle

from datasets.imdb import imdb
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

class datarr(imdb):
  def __init__(self, name, nof_ent_classes, nof_rel_classes, image_set):
    
    imdb.__init__(self, name)
    self._image_set = image_set
    self._devkit_path = self._get_default_path()
    
    self.set_paths()

    self._classes = ["invalid", "subject", "object", "none"]
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._data = self._load_data()
    self._image_index = [i for i in self._data]
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'
    self.nof_ent_classes = nof_ent_classes
    self.nof_rel_classes = nof_rel_classes


    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)
  
  def set_paths(self):
    pass

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, index)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _get_widths(self):
    return [self.im_metadata[m]["width"] for m in self.im_metadata]

  def _get_default_path(self):
    """
    Return the default path where vg is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, self.name)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    gt_roidb = [self._load_annotation(index)
                for index in self.image_index]


    return gt_roidb

  def rpn_roidb(self):
    roidb = self.gt_roidb()
    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_annotation(self, index):
    """
    Load image and bounding boxes info
    """
    image = index
    metadata = self.im_metadata[image]
    anot = self.annotations[image]
    width = metadata['width']
    height = metadata['height']
    
    seen_objs = {}
    for _, relation in enumerate(anot):
      sub = relation["subject"]
      if not str(sub) in seen_objs:
          seen_objs[str(sub)] = sub
      
      obj = relation["object"] 
      if not str(obj) in seen_objs:
          seen_objs[str(obj)] = obj
    num_objs = len(seen_objs)

    boxes = np.zeros((num_objs, 4), dtype=np.float32)
    partial_entity_class = np.zeros((num_objs, self.nof_ent_classes), dtype=np.int32)
    partial_relation_class = np.zeros((num_objs, num_objs, self.nof_rel_classes), dtype=np.int32)
    
    gt_classes = np.zeros((0, num_objs, 1), dtype=np.int32)
    overlaps = np.zeros((0, num_objs, self.num_classes), dtype=np.int64)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    queries = np.zeros((0, 2 * self.nof_ent_classes + self.nof_rel_classes), dtype=np.float32)
    # Load object bounding boxes into a data frame.    
    one_hot_obj = np.eye(self.nof_ent_classes) 
    for ix, obj in enumerate(seen_objs):
      boxes[ix] = np.array(seen_objs[obj]["bbox"])[[2,0,3,1]]
      partial_entity_class[ix] = one_hot_obj[seen_objs[obj]["category"]]
      seg_areas[ix] = (boxes[ix][2] - boxes[ix][0]) * (boxes[ix][3] - boxes[ix][1]) 
    
    indices = np.where(boxes[:, 2].astype(int) == boxes[:, 0].astype(int))
    boxes[indices ,2] += 1
    indices = np.where(boxes[:, 3].astype(int) == boxes[:, 1].astype(int))
    boxes[indices ,3] += 1

    assert (boxes[:, 2] > boxes[:, 0]).all()
    assert (boxes[:, 3]	 > boxes[:, 1]).all()  
    
    #load gt classe
    seen_rel = {}
    one_hot_rel = np.eye(self.nof_rel_classes)
    for _, relation in enumerate(anot):
        sub = relation["subject"]
        obj = relation["object"]
        sub_index = seen_objs.keys().index(str(sub))
        obj_index = seen_objs.keys().index(str(obj))        
        partial_relation_class[sub_index, obj_index, relation["predicate"]] = 1
    

    for _, relation in enumerate(anot):       
        sub = relation["subject"]
        obj = relation["object"]
        sub_index = seen_objs.keys().index(str(sub))
        obj_index = seen_objs.keys().index(str(obj)) 
        if sub_index == obj_index:
            continue

        sub_class = sub["category"]
        obj_class = obj["category"]
        rel_class = relation["predicate"]
        rel_str = str(sub_class)  + "_" + str(rel_class) + "_" + str(obj_class)
        found = False
        if not rel_str in seen_rel:
            query_gt_classes = np.zeros((1, num_objs, 1), dtype=np.int32)
            query_overlaps = np.zeros((1, num_objs, self.num_classes), dtype=np.int64)
            query_overlaps[0, :, 3] = 1
            query_gt_classes[0, :, 0] = 3
            seen_rel[rel_str] = 1
            
            for sub_ix, sub in enumerate(seen_objs):
                if seen_objs[sub]["category"] != sub_class:
                    continue
                for obj_ix, obj in enumerate(seen_objs):
                    if obj_ix == sub_ix:
                        continue   
                    if seen_objs[obj]["category"] != obj_class:
                        continue
                    if partial_relation_class[sub_ix, obj_ix, rel_class] == 1:         
                        query_gt_classes[0, sub_ix, 0] = 1
                        query_overlaps[0, sub_ix, 1] = 1
                        query_overlaps[0, sub_ix, 3] = 0
                        query_gt_classes[0, obj_ix, 0] = 2
                        query_overlaps[0, obj_ix, 2] = 1
                        query_overlaps[0, obj_ix, 3] = 0
                        found = True
            
            assert found
            gt_classes = np.concatenate((gt_classes, query_gt_classes), axis=0)
            overlaps = np.concatenate((overlaps, query_overlaps), axis=0)
            query = np.concatenate((one_hot_obj[sub_class], one_hot_rel[rel_class], one_hot_obj[obj_class]), axis=0)
            query = query.reshape([1, -1])
            queries = np.concatenate((queries, query), axis=0)
            
    # for the partial sence-graph training - use single relation
    for sub_ix, sub in enumerate(seen_objs):
        for obj_ix, obj in enumerate(seen_objs):
            relation = partial_relation_class[sub_ix, obj_ix]
            sumrel = np.sum(relation)
            if sumrel > 1:
                index = np.random.choice(relation.shape[0], 1, p=relation/sumrel)
                partial_relation_class[sub_ix, obj_ix] = np.zeros_like(partial_relation_class[sub_ix, obj_ix])
                partial_relation_class[sub_ix, obj_ix, index] = 1

    nof_relations = np.sum(partial_relation_class, axis=2)
    assert (nof_relations <= 1).all()

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
            'query' : queries,
            'partial_entity_class' : partial_entity_class,
            'partial_relation_class' : partial_relation_class,
            'orig_image': None}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
      pass
  def _do_python_eval(self, output_dir='output'):
    pass

  def _do_matlab_eval(self, output_dir='output'):
      pass

  def evaluate_detections(self, all_boxes, output_dir):
      pass
  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def _load_data(self):
    print("start loading data " + self._image_set)
    # read json file
    self.im_metadata = json.load(open(self._im_metadata_path))
    self.annotations = json.load(open(self._annotations_path))
    print("done loading data " + self._image_set)
    return [i for i in self.im_metadata if i in self.annotations] 
