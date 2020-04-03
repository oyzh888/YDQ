import os


data_root = '/Users/mac/Downloads/PoseEstimation/'
video_root = os.path.join(data_root, 'video')
json_root = os.path.join(data_root, 'HRNet')
# json_root = os.path.join(data_root, 'Lightweight\ OpenPose')
out_root = os.path.join(data_root, 'output')

if not os.path.exists(out_root): os.mkdir(out_root)

file_list = [
    '1.侧身深蹲',
    '3.侧身俄罗斯转体',
    '4.俯身转体',
    '6.开合跳'
]

# filter_parameters = [15, 25, 35]
filter_parameters = [15, 25, 35]
mark = 'HRNet_25'

for f in file_list:
    json_file = os.path.join(json_root,f+'.json')
    video_file = os.path.join(video_root, f + '.mp4')
    output_file = os.path.join(out_root, f + f'_{mark}.mp4')
    cmd = f'python smooth_json.py --video \'{video_file}\' --json \'{json_file}\' --output \'{output_file}\''
    # print(cmd)
    os.system(f'python smooth_json.py --video {video_file} --json {json_file} --output {output_file}')
    # exit()