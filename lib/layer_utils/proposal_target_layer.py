# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, gt_labels, ent_labels, rel_labels, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    #all_rois = np.vstack(
    #  (all_rois, np.hstack((zeros, gt_boxes)))
    #)
    all_rois = np.hstack((zeros, gt_boxes))
    # not sure if it a wise appending, but anyway i am not using it
    #all_scores = np.vstack((all_scores, zeros))
    all_scores = zeros
  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

  # Sample rois with classification labels and bounding box regression
  # targets
  ent_labels, rel_labels, labels, rois, roi_scores, bbox_targets, bbox_inside_weights, labels_mask = _sample_rois(
    all_rois, all_scores, gt_boxes, gt_labels, ent_labels, rel_labels, fg_rois_per_image,
    rois_per_image, _num_classes)

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(labels.shape[0], labels.shape[1], 1).astype(np.float32)
  bbox_targets = bbox_targets.reshape(-1, 4).astype(np.float32)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, 4).astype(np.float32)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, ent_labels, rel_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, labels_mask


def _get_bbox_regression_labels(bbox_target_data, fg_inds):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = np.ones(bbox_target_data.shape[0])
  bbox_targets = np.zeros((clss.size, 4), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  bbox_targets = bbox_target_data
  bbox_inside_weights[fg_inds] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return targets

def _sample_rois(all_rois, all_scores, gt_boxes, gt_labels, ent_labels, rel_labels, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:,1:5], dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
  gt_assignment = overlaps.argmax(axis=1)
  max_overlaps = overlaps.max(axis=1)
  labels = gt_labels[:, gt_assignment, 0]
  ent_labels = ent_labels[gt_assignment]
  rel_labels = rel_labels[gt_assignment, :, :][:, gt_assignment, :]

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  labels_mask = np.zeros_like(max_overlaps ,dtype=np.float32)
  labels_mask[fg_inds] = 1.0


  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
  labels[:, bg_inds] = 0
  labels_mask[bg_inds] = - 1.0

  rois = all_rois
  roi_scores = all_scores
  gt_boxes = gt_boxes[gt_assignment]
  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, fg_inds)

  return ent_labels, rel_labels, labels, rois, roi_scores, bbox_targets, bbox_inside_weights, labels_mask #gt_assignment[keep_inds]
