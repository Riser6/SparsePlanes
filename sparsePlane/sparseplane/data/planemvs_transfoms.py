import copy
import numpy as np
import os
import torch
import pickle
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)
import pycocotools.mask as mask_util
from PIL import Image
import torchvision.transforms as transforms
from . import GaussianBlur

__all__ = ["PlaneMVSMapper"]

def load_instances(
    dataset_dict, image_size, mask_format="polygon", max_num_planes=20
):
    target = Instances(image_size)

    """segmentation =  cv2.imread(dataset_dict["plane_segment_path"], -1).astype(np.int32)
    segmentation = (segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]) // 100 - 1
    segments, counts = np.unique(segmentation, return_counts=True)
    segmentList = zip(segments.tolist(), counts.tolist())
    segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
    segmentList = sorted(segmentList, key=lambda x:-x[1])

    newPlanes = []
    newPlaneInfo = []
    newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)

    newIndex = 0
    for oriIndex, count in segmentList:
        if count < 50000:
            continue
        if oriIndex >= len(planes):
            continue            
        if np.linalg.norm(planes[oriIndex]) < 1e-4:
            continue
        newPlanes.append(planes[oriIndex])
        newSegmentation[segmentation == oriIndex] = newIndex
        newPlaneInfo.append(plane_info[oriIndex] + [oriIndex])
        newIndex += 1
        continue

    segmentation = newSegmentation
    planes = np.array(newPlanes)
    plane_info = newPlaneInfo

    confident_labels = {1: True, 2: True, 3: True, 4: True, 7: True, 8: True, 9: True, 11: True, 12: True, 14: True, 17: True, 19:True, 20:True, 22: True, 24:True, 25:True, 29: True, 30: True, 32:True}

    if len(planes) > 0:
        planes = transformPlanes(extrinsics, planes)
        segmentation, plane_depths = cleanSegmentation(image, planes, plane_info, segmentation, depth, camera, planeAreaThreshold=500, planeWidthThreshold=10, confident_labels=confident_labels, return_plane_depths=True)
        
        masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
        plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
        plane_mask = masks.max(2)
        plane_mask *= (depth > 1e-4).astype(np.float32)            
        plane_area = plane_mask.sum()
        depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)
        if depth_error > 0.1:
            print('depth error', depth_error)
            planes = []"""




class PlaneMVSMapper:
    def __init__(self, cfg, is_train=True, dataset_names=None):
        self.cfg = cfg
        self.depth_on       = cfg.MODEL.DEPTH_ON
        self._augmentation  = cfg.DATALOADER.AUGMENTATION

        self.is_train = is_train
        
        assert dataset_names is not None
        if self._augmentation:
            color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            augmentation = [
                transforms.RandomApply([color_jitter], p=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
            ]
            self.img_transform = transforms.Compose(augmentation)
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        for i in range(2):
            image = cv2.imread(dataset_dict[str(i)]["img_path"])
            image = cv2.resize(image, (640, 480))
            if self.is_train and self._augmentation:
                image = Image.fromarray(image)
                dataset_dict[str(i)]["image"] = self.img_transform(image) * 255.0
                image_shape = dataset_dict[str(i)]["image"].shape[1:]
            else:
                dataset_dict[str(i)]["image"] = torch.as_tensor(
                    image.transpose(2, 0, 1).astype("float32")
                )
            if self.depth_on:
                if "depth_head" in self.cfg.MODEL.FREEZE:
                    dataset_dict[str(i)]["depth"] = torch.as_tensor(
                        np.zeros((480, 640)).astype("float32")
                    )
                else:
                    dataset_dict[str(i)]["depth"] = torch.as_tensor(
                        cv2.imread(dataset_dict[str(i)]["depth_path"], -1).astype(np.float32) / self.depthShift
                    )
            
        if not self.is_train and not self._eval_gt_box:
            return dataset_dict
            
        if not self._eval_gt_box:
            for i in range(2):
                instances = load_instances(dataset_dict[str(i)], image_shape)
                dataset_dict[str(i)]["instances"] = instances[
                    instances.gt_boxes.nonempty()
                ]
        else:
            for i in range(2):
                pass    #todo
            

            