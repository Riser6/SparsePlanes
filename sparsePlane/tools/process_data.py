from email.errors import NonPrintableDefect
from linecache import lazycache
import logging
import os
import json
import numpy as np
import glob
from pathlib import Path

import os
import cv2

#data_root = "/mnt/cache/wudong/Droid_Plane/scannet"

data_root = "/data/wudong/SparsePlanes/sparsePlane/datasets/scannet"


frameGap = 5
keyframeGap = 1

classes_mapping = {41:  17,   #background
                    1:  1,   #wall
                    2:  2,   #floor
                    20: 2,   #floor mat
                    3:  3,   #cabinet
                    12: 3,   #counter
                    17: 3,   #dresser
                    32: 3,   #night stand
                    4:  4,   #bed
                    7:  5,   #table
                    14: 5,   #desk
                    8:  6,   #door
                    9:  7,   #windows
                    13: 7,   #blinds
                    11: 8,   #picture
                    22: 9,   #ceiling
                    29: 10,  #box
                    30: 11,  #whiteboard
                    24: 12,  #refrigerator
                    25: 13,  #tv
                    5:  14,  #chair
                    6:  15,  #sofa
                    10: 16,  #bookshelf
                    15: 16,  #shelf
                    16: 17,  #curtain
                    18: 17,  #pillow
                    19: 17,  #mirror
                    21: 17,  #clothes
                    23: 17,  #books
                    26: 17,  #paper
                    27: 17,  #towel
                    28: 17,  #shower curtain
                    31: 17,  #person
                    33: 17,  #toilet
                    34: 17,  #sink
                    35: 17,  #lamp
                    36: 17,  #bathtub
                    37: 17,  #bag
                    38: 17,  #otherstructure
                    39: 17,  #otherfurniture
                    40: 17}  #otherprop


def parse_camera_info(folder):
    with open(folder) as f:
        for line in f:
            line = line.strip()
            tokens = [token for token in line.split(' ') if token.strip() != '']
            if tokens[0] == "fx_depth":
                fx = float(tokens[2])
            if tokens[0] == "fy_depth":
                fy = float(tokens[2])
            if tokens[0] == "mx_depth":
                cx = float(tokens[2])                            
            if tokens[0] == "my_depth":
                cy = float(tokens[2])
            elif tokens[0] == "depthWidth":
                Width = int(tokens[2])
            elif tokens[0] == "depthHeight":
                Height = int(tokens[2])
                pass
    assert Width == 640 and Height == 480
    return fx, fy, cx, cy, Width, Height

def load_camera_pose(camera_pose_file):
    camera_pose_inv = np.loadtxt(camera_pose_file)
    camera_pose = np.linalg.inv(camera_pose_inv)
    T = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    return np.dot(T, camera_pose) 

def transformPlanes(transformation, planes):
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        
    centers = planes
    centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
    newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
    newCenters = newCenters[:, :3] / newCenters[:, 3:4]

    refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
    refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
    newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
    newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

    planeNormals = newRefPoints - newCenters
    planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
    planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
    newPlanes = planeNormals * planeOffsets
    return newPlanes

def calcPlaneDepths(planes, width, height, camera, max_depth=10):
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
    
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    
    return planeDepths

def cleanSegmentation(image, planes, plane_info, segmentation, depth, camera, planeAreaThreshold=200, planeWidthThreshold=10, depthDiffThreshold=0.1, validAreaThreshold=0.5, brightThreshold=20, confident_labels={}, return_plane_depths=False):

    planeDepths = calcPlaneDepths(planes, segmentation.shape[1], segmentation.shape[0], camera).transpose((2, 0, 1))
    
    newSegmentation = np.full(segmentation.shape, fill_value=-1)
    validMask = np.logical_and(np.linalg.norm(image, axis=-1) > brightThreshold, depth > 1e-4)
    depthDiffMask = np.logical_or(np.abs(planeDepths - depth) < depthDiffThreshold, depth < 1e-4)

    for segmentIndex in np.unique(segmentation):
        if segmentIndex < 0:
            continue
        segmentMask = segmentation == segmentIndex

        try:
            plane_info[segmentIndex][0][1]
        except:
            print('invalid plane info')
            print(plane_info)
            print(len(plane_info), len(planes), segmentation.min(), segmentation.max())
            print(segmentIndex)
            print(plane_info[segmentIndex])
            exit(1)

        if plane_info[segmentIndex][0][1] in confident_labels:
            if segmentMask.sum() > planeAreaThreshold:
                newSegmentation[segmentMask] = segmentIndex
                pass
            continue
        oriArea = segmentMask.sum()
        segmentMask = np.logical_and(segmentMask, depthDiffMask[segmentIndex])
        newArea = np.logical_and(segmentMask, validMask).sum()
        if newArea < oriArea * validAreaThreshold:
            continue
        segmentMask = segmentMask.astype(np.uint8)
        segmentMask = cv2.dilate(segmentMask, np.ones((3, 3)))
        numLabels, components = cv2.connectedComponents(segmentMask)
        for label in range(1, numLabels):
            mask = components == label
            ys, xs = mask.nonzero()
            area = float(len(xs))
            if area < planeAreaThreshold:
                continue
            size_y = ys.max() - ys.min() + 1
            size_x = xs.max() - xs.min() + 1
            length = np.linalg.norm([size_x, size_y])
            if area / length < planeWidthThreshold:
                continue
            newSegmentation[mask] = segmentIndex
            continue
        continue
    if return_plane_depths:
        return newSegmentation, planeDepths
    return newSegmentation

def load_plane(semantic_segment_path, plane_segment_path, planes, plane_info, camera_pose, camera, camera_info, image_path, depth_path):
    image = cv2.imread(image_path)
    depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000
    image = cv2.resize(image, (depth.shape[1], depth.shape[0]))

    segmentation =  cv2.imread(plane_segment_path, -1).astype(np.int32)
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
        planes = transformPlanes(camera_pose, planes)
        segmentation, plane_depths = cleanSegmentation(image, planes, plane_info, segmentation, depth, camera_info, planeAreaThreshold=500, planeWidthThreshold=10, confident_labels=confident_labels, return_plane_depths=True)

        masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
        plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
        plane_mask = masks.max(2)
        plane_mask *= (depth > 1e-4).astype(np.float32)            
        plane_area = plane_mask.sum()
        depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)
        if depth_error > 0.1:
            print('depth error', depth_error)
            planes = []
    if len(planes) == 0 or segmentation.max() < 0:
        return -1
    
    class_ids = []        
    parameters = []
    semantic_ids = []

    plane_offsets = np.linalg.norm(planes, axis=-1)                    
    plane_normals = planes / np.expand_dims(plane_offsets, axis=-1)
    distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
    normal_anchors = distances_N.argmin(-1)
    
    new_planeindex = 0
    for planeIndex, plane in enumerate(planes):
        m = segmentation == planeIndex
        if m.sum() < 1:
            segmentation[m] = -1
            continue
        segmentation[m] = new_planeindex

        class_ids.append(normal_anchors[planeIndex] + 1)
        normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
        parameters.append(np.concatenate([normal, np.array([plane_info[planeIndex][-1]])], axis=0))
        semantic_ids.append(classes_mapping[plane_info[planeIndex][0][1]])
        new_planeindex += 1
    
def get_frame_info(scenePath, scene_id, frame_id, droid_poses, camera_info, planes, plane_info):
    frame_info = {}
    frame_info["img_path"] = os.path.join(scenePath, "color", str(frame_id)+".jpg")
    frame_info["depth_path"] = os.path.join(scenePath, "depth", str(frame_id)+".png")
    frame_info["droid_depth"] = os.path.join(scenePath, "depthdroidgrays1sm", str(frame_id)+".txt")
    """frame_info["semantic_segment_path"] = os.path.join(scenePath, "label-filt", str(frame_id)+".png")
    frame_info["plane_segment_path"] = os.path.join(scenePath, "annotation/segmentation", str(frame_id)+".png")"""
    
    frame_info["image_id"] = scene_id
    frame_info["height"] = 480
    frame_info["weight"] = 640

    camera_pose = load_camera_pose(os.path.join(scenePath, 'pose', str(frame_id)+".txt"))
    frame_info["camera_pose"] = camera_pose.tolist()
    frame_info["droid_pose"] = droid_poses[frame_id].tolist()
    
    semantic_segment_path = os.path.join(scenePath, "label-filt", str(frame_id)+".png")
    plane_segment_path = os.path.join(scenePath, "annotation/segmentation", str(frame_id)+".png")
    
    plane_segmentes, plane_classes = load_plane(semantic_segment_path, plane_segment_path, planes, plane_info, camera_pose, camera_info, frame_info["img_path"], frame_info["depth_path"])

    return frame_info


def process_data(data_root, split):
    scannet_data = []
    planenet_scene_ids_val = np.load(os.path.join(data_root, 'scene_ids_val.npy'))
    planenet_scene_ids_val = {scene_id.decode('utf-8'): True for scene_id in planenet_scene_ids_val}
    invalid_indices = {}

    # Remove the average depth discrepancy over planar regions is larger than 0.1m
    with open(data_root + '/invalid_indices_' + split + '.txt', 'r') as f:
        for line in f:
            tokens = line.split(' ')
            if len(tokens) == 3:
                invalid_index = int(tokens[1]) * 10000 + int(tokens[2])
                if invalid_index not in invalid_indices:
                    invalid_indices[invalid_index] = True
    
    # Remove scenes without droid_depth(too long, memory out)
    invalid_scene_ids = []
    with open(data_root + "/invalid_data.txt") as f:
            for line in f:
                invalid_scene_ids.append(line.replace('\n', ''))

    scene_idx = -1
    with open(data_root + '/ScanNet/Tasks/Benchmark/scannetv1_' + split + '.txt') as f:
        for line in f:
            scene_id = line.strip()
            if split == 'test':
                if scene_id not in planenet_scene_ids_val:
                     continue
            scene_idx += 1

            if scene_id in invalid_scene_ids:
                continue
            
            scenePath = data_root + '/scans/' + scene_id
            if not os.path.exists(scenePath + '/' + scene_id + '.txt') or not os.path.exists(scenePath + '/annotation/planes.npy'):
                continue
            
            camera_info = parse_camera_info(scenePath + '/' + scene_id + '.txt')
            planes = np.load(scenePath + '/annotation/planes.npy', allow_pickle=True)
            plane_info = np.load(scenePath + '/annotation/plane_info.npy', allow_pickle=True)
            keyframe_ids = sorted([int(Path(filename).stem) for filename in glob.glob(scenePath + '/depthdroidgrays1/*.txt')])
            droid_poses = np.load(scenePath + '/posedroid/result_poses.npy')
            droid_poses = None

            for idx, keyframe_id in enumerate(keyframe_ids):
                former_idx = idx - keyframeGap
                if (scene_idx * 10000 + keyframe_id) in invalid_indices:
                    continue

                while former_idx >= 0:
                    if keyframe_ids[former_idx] + frameGap < keyframe_id:
                        break
                    former_idx -= 1

                keyframe_info =  get_frame_info(scenePath, scene_id, keyframe_id, droid_poses, camera_info, planes, plane_info)

                if former_idx >=0:
                    former_id = keyframe_ids[former_idx]
                    former_frame_info = get_frame_info(scenePath, former_id, former_id, droid_poses, camera_info, planes, plane_info)
                    scannet_data.append({0: keyframe_info, 1:former_frame_info})
    return scannet_data


if __name__ == "__main__":
    process_data(data_root, "train")