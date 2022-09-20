import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.interpolate as interp

class DataLoaderDab:
    def __init__(self, BASE_DIR='./hardness'):
        self.BASE_DIR = BASE_DIR
        self.scaler = StandardScaler()

    def interpolate(self, arr, interp_to_frame_len, method='linear'):
        n_frames, n_dim = arr.shape
        interp_arr = np.zeros((interp_to_frame_len, arr.shape[-1]))
        if method == 'linear':
            for i in range(n_dim):
                arr_interp = interp.interp1d(np.arange(arr.shape[0]), arr[:, i])
                interp_arr[:, i] = arr_interp(np.linspace(0, arr.shape[0] - 1, interp_to_frame_len))
            return interp_arr
        elif method == 'cubic':
            for i in range(n_dim):
                arr_interp = interp.interp1d(np.arange(arr.shape[0]), arr[:, i], kind='cubic')
                interp_arr[:, i] = arr_interp(np.linspace(0, arr.shape[0] - 1, interp_to_frame_len))
            return interp_arr
        elif method == 'nearest':
            pass

    #the input array contains [accX, accY, accZ, P1, P2, P3]
    def transform_axes(self, arr):
        acc = arr[:, :3]
        p = arr[:, 3:]
        rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # rot_mat = np.identity(3)
        acc_T = np.matmul(rot_mat, acc.T).T
        p_T = np.zeros(p.shape)
        p_T[:, 0] = (p[:, 0] - p[:, 1]) * np.cos(np.pi/6)
        p_T[:, 1] = ((p[:, 0] + p[:, 1]) * np.cos(np.pi/3)) - p[:, 2]
        p_T[:, 2] = (p[:, 0] + p[:, 1] + p[:, 2]) / 3
        arr_T = np.hstack((acc_T, p_T))
        # arr_T_N = self.scaler.fit_transform(arr_T)
        return arr_T



    def split_sweep_into_windows(self, arr, window_size, label, axis_selection_tuple):
        n_samples, n_dim = arr.shape
        window_arr = np.zeros((n_samples//window_size, window_size, len(axis_selection_tuple)))
        window_label_list = [label]*(n_samples//window_size)
        for start_index in np.arange(0, (n_samples//window_size)*window_size, window_size):
            window_arr[start_index//window_size] = arr[start_index:start_index+window_size][:, axis_selection_tuple]
        return window_arr, window_label_list



    def load_full_data_with_labels(self, axis_selection_tuple=(0, 1, 2, 3, 4, 5),
                                   full_data_norm=False, downsample_factor=1, window_size=50):

        arr_list = []
        label_list = []
        for file_name in os.listdir(self.BASE_DIR):
            fpath = os.path.join(self.BASE_DIR, file_name)
            df = pd.read_csv(fpath)
            data_arr = df.to_numpy()
            n_samples, _ = data_arr.shape
            data_arr = self.interpolate(data_arr, interp_to_frame_len=n_samples // downsample_factor)
            data_arr_T = self.transform_axes(data_arr)
            split_window_arr, window_split_label_list = self.split_sweep_into_windows(data_arr_T,
                                                                                      window_size=window_size,
                                                                                      axis_selection_tuple=axis_selection_tuple,
                                                                                      label=file_name)
            arr_list.append(split_window_arr)
            label_list = label_list + window_split_label_list
        return np.concatenate(arr_list), pd.factorize(np.vstack(label_list)[:, 0])[0], pd.factorize(np.vstack(label_list)[:, 0])[1], np.unique(np.vstack(label_list)[:, 0])
