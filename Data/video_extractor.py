# Importing all necessary libraries
import cv2
import os

# init path, then run
data_root = "/Users/mac/Downloads/Examples-video/"
video_folder_path = os.path.join(data_root, "./")
output_folder_path = os.path.join(data_root, "output")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def save_video(video_path, video_out_path):
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)

    # frame
    currentframe = 0

    # init video outpath folder
    if not os.path.exists(video_out_path):
        os.makedirs(video_out_path)

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = os.path.join(video_out_path, str(currentframe) + '.jpg')
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

for file in os.listdir(video_folder_path):
    if not os.path.isdir(file):#判断是否是文件夹，不是文件夹才打开
        # print(file)
        video_path = os.path.join(video_folder_path, file)
        save_video(video_path, os.path.join(output_folder_path, file[:-4]))
        break