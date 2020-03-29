import argparse
import cv2
import json
import numpy as np


BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]


class Pose(object):
    num_kpts = 18
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    colors = [[0, 224, 255], [84, 2, 253], [255, 0, 128]]

    def __init__(self, keypoints, confidence, kp_threshold=-0.0001):
        super().__init__()
        kp_info = np.array(keypoints).reshape(-1, 3) #keypoints的格式是[x_0, y_0, c_0, x_1, y_1, c_1, ...]，所以reshape成三个一行，前两点是坐标，后面一个点是score
        keypoints = kp_info[:, 0:2].astype(np.long)
        scores = kp_info[:, 2]
        # if the score of a keypoint is small, we eliminate this keypoint
        for idx, score in enumerate(scores):
            if score <= kp_threshold:
                keypoints[idx, 0] = -1
                keypoints[idx, 1] = -1
        self.keypoints = keypoints
        self.confidence = confidence

    def draw(self, img, ids=0):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        # len(BODY_PARTS_PAF_IDS) - 2 = 19 - 2
        for part_id in range(17):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.colors[ids], -1) #img为要画的图，其后是圆心，接着3是半径，然后是颜色和样式的选择
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.colors[ids], -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.colors[ids], 2)
    def draw_hrnet(self, img, ids=0):
        assert self.keypoints.shape == (Pose.num_kpts, 2)
        neck_x = (self.keypoints[2][0]+self.keypoints[5][0])/2
        neck_y = (self.keypoints[2][1]+self.keypoints[5][1])/2
        # len(BODY_PARTS_PAF_IDS) - 2 = 19 - 2
        for part_id in range(17):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.colors[ids], -1) #img为要画的图，其后是圆心，接着3是半径，然后是颜色和样式的选择
            else:
                x_a, y_a = neck_x, neck_y
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.colors[ids], -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.colors[ids], -1)
            else:
                x_a, y_a = neck_x, neck_y
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.colors[ids], -1)
            if global_kpt_a_id != -1 or global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.colors[ids], 2)


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.count = -1
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self): #前后加横线，代表是调用iter，此处的class是定义了一个迭代器，可以用next()来读取每一帧
        self.cap = cv2.VideoCapture(self.file_name)#读取视频
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        self.count += 1
        was_read, img = self.cap.read()#从视频或摄像头中读取一帧（即一张图像）,返回是否成功标识was_read(True代表成功，False代表失败），img为读取的视频帧
        if not was_read:
            raise StopIteration
        return img, self.count


def plotter(frame_provider, total_keypoints, output_file, fps):
    assert len(total_keypoints) > 0
    preds = list(total_keypoints.keys())
    prefix = '_'.join(preds[0].split('_')[:-1])
    preds = [int(pred.split('_')[-1]) for pred in preds]
    #preds = [int(pred.split('.')[0]) for pred in preds] #for tensforlow

    cur_pred = -1
    video_writer = None
    for img, image_idx in frame_provider:
        if video_writer is None:
            h, w, c = img.shape
            #video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))#创建视频流写入对象，VideoWriter_fourcc为视频编解码器，fps帧播放速率，（w，h）为视频帧大小
            video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
        if image_idx in preds:
            cur_pred = image_idx

        pose_entries = total_keypoints['{}_{}'.format(prefix, cur_pred)] #选取json中有被读到帧的图片拿来画图，没读到的就沿用上一帧读到的图片
        #pose_entries = total_keypoints['{}.jpg'.format(cur_pred)]  #for tensforlow
        for n in range(len(pose_entries)):
            pose_keypoints = pose_entries[n]['keypoints']
            pose_score = pose_entries[n]['score']

            pose = Pose(pose_keypoints, pose_score)
            #pose.draw(img, ids=n%3) 
            pose.draw_hrnet(img, ids=n%3) #for hrnet

        video_writer.write(img)#向视频文件写入一帧
    video_writer.release() #释放视频流



def Splicing(video1,video2,video3,output_file): 


    video1 = cv2.VideoCapture(video1)
    video2= cv2.VideoCapture(video2)
    video3= cv2.VideoCapture(video3)


    fps = video1.get(cv2.CAP_PROP_FPS)

    width = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width*3, height))

    success1, frame1 = video1.read()
    success2, frame2 = video2.read()
    success3, frame3 = video3.read()
  

    while success1 and success2 and success3 :
       
        model_list=['tensorflow','openpose','HRnet']
    	left_x_up=0
    	left_y_up=0
    	right_x_down = int(left_x_up + width / 10)
    	right_y_down = int(left_y_up + height / 10)
    	word_x = left_x_up + 5
    	word_y = left_y_up + 25
    	cv2.putText(frame1, model_list[0], (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX ,1, (0, 224, 255), 2)
    	cv2.putText(frame2, model_list[1], (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 224, 255), 2)
    	cv2.putText(frame3, model_list[2], (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 224, 255), 2)
    	frame1 = cv2.resize(frame1,(width , height), interpolation=cv2.INTER_CUBIC)
    	frame2 = cv2.resize(frame2, (width , height), interpolation=cv2.INTER_CUBIC)
   		frame3= cv2.resize(frame3, (width , height), interpolation=cv2.INTER_CUBIC)
        frame = np.hstack((frame1, frame2,frame3))


        videoWriter.write(frame)
        success1, frame1 = video1.read()
        success2, frame2 = video2.read()
        success3, frame3 = video3.read()
    

    videoWriter.release()
    video1.release()
    video2.release()
    video3.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', default='checkpoints/checkpoint_iter_370000.pth',
                        type=str, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file')
    parser.add_argument('--json', type=str, default='', help='path to json file')
    parser.add_argument('--output', type=str, default='output.mp4', help='path to video file')
    parser.add_argument('--fps', type=int, default=30, help='path to video file')
    args = parser.parse_args()

    args.video = 'example/basic_squat.mp4'
    args.json = 'example/basic_squat.json'
    if args.video == '' and args.json == '':
        raise ValueError('--video and --json have to be provided')
    frame_provider = VideoReader(args.video)

    with open(args.json, 'r') as result_file:
        total_keypoints = json.load(result_file)
        # print(total_keypoints)
        # assert False
    # print(len(total_keypoints[0]['data/basic_squat_frame_0000'][0]['keypoints']))
    # assert False

    plotter(frame_provider, total_keypoints, output_file=args.output, fps=args.fps)
