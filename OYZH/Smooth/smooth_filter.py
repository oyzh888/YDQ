import numpy as np
from scipy import signal

class Smooth_savgol_filter():
    def __init__(self, keypoints_dim = 18, filter_size = 25):
        # filter_size has to be an odd number for Smooth_savgol_filter
        print('init:', filter_size)
        if filter_size % 2 == 0: filter_size += 1
        self.keypoints_dim = keypoints_dim
        self.stack = np.empty((keypoints_dim * 2, 0), float)
        self.stack_len = filter_size + 1
        self.filter_size = filter_size

    def add_key_point(self, keypoints):

        # print('input:')
        # print(keypoints, keypoints.shape, type(keypoints))
        xy_array = np.append(keypoints[:, 0], keypoints[:,1])
        xy_array = np.expand_dims(xy_array, axis=1)
        self.stack = np.concatenate((self.stack, xy_array), axis=1)

        if self.stack.shape[1] > self.stack_len:
            # self.stack.pop()
            self.stack = self.stack[:,1:]
            new_res = np.empty(shape = self.stack.shape)
            for idx, row in enumerate(self.stack):
                smooth_row = self.smooth_data(row)
                new_res[idx,:] = smooth_row
            return_res = np.array([new_res[0:self.keypoints_dim,-1], new_res[self.keypoints_dim: 2*self.keypoints_dim,-1]]).T

            return_res = return_res.astype(int)
            # delete invalid result:
            return_res[return_res == 0] = -1
            # print('return_res')
            # print(return_res, return_res.shape, type(return_res))
            return return_res
        else:
            return keypoints

    def smooth_data(self, data):
        return signal.savgol_filter(data, window_length=self.filter_size, polyorder=2)

global_filter_size = 25
smooth_filter = Smooth_savgol_filter(filter_size = global_filter_size)


keypoints = np.array([[213, 102],[ -1,  -1],[276, 186], [283, 322],[288, 449],  [287, 193], [300, 325], [321, 436], [303, 413]
             ,[303, 586], [306, 711], [288, 419], [309, 611], [309, 788], [221, 86], [224, 89], [269,  90],  [264, 97]])
new_keypoints = smooth_filter.add_key_point(keypoints)
print(new_keypoints)