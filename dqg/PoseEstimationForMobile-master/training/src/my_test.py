import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os
import math
import time

from scipy.ndimage.filters import gaussian_filter
from scipy import spatial
from IPython import embed

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)


def cal_coord(pred_heatmaps, images_size):
    coords = {}
    for img_id in pred_heatmaps.keys():
        heat_h, heat_w, n_kpoints = pred_heatmaps[img_id].shape
        scale_h, scale_w = heat_h / images_size[img_id][0], heat_w / images_size[img_id][1]
        coord = []
        for p_ind in range(n_kpoints):
            heat = pred_heatmaps[img_id][:, :, p_ind]
            heat = gaussian_filter(heat, sigma=5)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            coord_x = int((ind[1] + 1) / scale_w)
            coord_y = int((ind[0] + 1) / scale_h)
            coord.append((coord_x, coord_y))
        coords[img_id] = coord
    return coords


def infer(frozen_pb_path, output_node_name, img_path):
    with tf.gfile.GFile(frozen_pb_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name("image:0")
    output_heat = graph.get_tensor_by_name("%s:0" % output_node_name)

    res = {}
    use_times = []
    images_size = {}
    with tf.Session() as sess:
        
        for file_name in os.listdir(img_path):
            img_id = int(file_name.split('.')[0])
            img_file_path = os.path.join(img_path, file_name)
            ori_img = cv2.imread(img_file_path)
            print(img_file_path)
            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(ori_img, (shape[1], shape[2]))
            st = time.time()
            heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            infer_time = 1000 * (time.time() - st)
            print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
            res[img_id] = np.squeeze(heat)
            images_size[img_id] = [ori_img.shape[0], ori_img.shape[1]]

    print("Average inference time = %.2f ms" % np.mean(use_times))
    return res, images_size

def normalize_coords(coords):
    sum_sq = 0.
    for i in range(len(coords)):
        sum_sq += coords[i] ** 2

    coords /= sum_sq 
    return coords

def clip_coords(coords):
    coords[:,0] -= np.min(coords[:,0])
    coords[:,1] -= np.min(coords[:,1])
    return coords

def pose_match(pred_heatmap, pred_coords, src_iter, tgt_itr):
    conf_score = []
    src_pos = []
    tgt_pos = []

    src_coords = clip_coords(np.array(pred_coords[src_iter], dtype=float))
    tgt_coords = clip_coords(np.array(pred_coords[tgt_iter], dtype=float))

    for i in range(14):
        conf_score.append(np.max(pred_heatmap[src_iter][:,:,i]))
        src_pos.append(src_coords[i][0])
        src_pos.append(src_coords[i][1])
        tgt_pos.append(tgt_coords[i][0])
        tgt_pos.append(tgt_coords[i][1])

    src_pos = normalize_coords(src_pos)
    tgt_pos = normalize_coords(tgt_pos)
    conf_score = np.array(conf_score, dtype=float)
    
    cos_dist = 1 - spatial.distance.cosine(src_pos, tgt_pos)
    return cos_dist

def save_json(json_name, pred_heatmap, pred_coords):
    keypoints_name_out = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"]
    keypoints_ori = ['Nose', 'Neck', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    keypoints_name = []
    transform = [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13]

    for t in transform:
        keypoints_name.append(keypoints_ori[t-1])

    json_dict = {}

    for i in range(len(pred_heatmap)):
        coords = pred_coords[i]
        heatmap = pred_heatmap[i]
        keypoints = {}
        for j in range(14):
            x = coords[j][0]
            y = coords[j][1]
            c = np.max(heatmap[:,:,j])
            keypoints[keypoints_name[j]] = [x, y, c]
    
        keypoints_out = []
        for keypoint_name_out in keypoints_name_out:
            if keypoint_name_out in keypoints.keys():
                keypoints_out.append(keypoints[keypoint_name_out][0])
                keypoints_out.append(keypoints[keypoint_name_out][1])
                keypoints_out.append(keypoints[keypoint_name_out][2])
            else:
                keypoints_out.append(-1.)
                keypoints_out.append(-1.)
                keypoints_out.append(0.)

        json_dict['{}.jpg'.format(i)] = [{'score':0.0, 'keypoints':[]}]
        json_dict['{}.jpg'.format(i)][0]['keypoints'] = keypoints_out

    with open('{}.json'.format(json_name),'w') as file:
        json.dump(json_dict, file, cls=MyEncoder)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="my benchmark") # begin with 235
    parser.add_argument("--frozen_pb_path", type=str, default="")
    parser.add_argument("--img_path", type=str, default="/home/dingqianggang/yd/yxy_dataset/yxy_example_video_dataset/5")
    parser.add_argument("--output_node_name", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--json_name", type=str, default="default")
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    pred_heatmap, images_size = infer(args.frozen_pb_path, args.output_node_name, args.img_path)
    pred_coords = cal_coord(pred_heatmap, images_size)

    save_json(args.json_name, pred_heatmap, pred_coords)

    src_range = np.arange(0, 235)
    tgt_range = np.arange(235, 301)

    for src_iter in src_range:
        scores = []
        ids = []
        for tgt_iter in tgt_range:
            score = pose_match(pred_heatmap, pred_coords, src_iter, tgt_iter)
            scores.append(score)
            ids.append(tgt_iter)

        sores = np.array(scores)
        max_score = np.max(scores)
        max_id = np.argmax(scores)
        max_tgt_id = ids[max_id]
        # print(scores)
        print(max_id, max_score)
        

    

