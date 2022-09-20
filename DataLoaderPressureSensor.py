import numpy as np
import os
import pandas as pd
# from promise.dataloader import DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.interpolate as interp

class DataLoaderPressureSensor:
    def __init__(self, BASE_DIR='./AUG03082022'):
        self.BASE_DIR = BASE_DIR
        self.scaler = StandardScaler()

    def interpolate(self, arr, interp_to_frame_len, method='linear'):
        n_frames, n_dim = arr.shape
        # print(n_frames, interp_to_frame_len)

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

    #three milling types ['HM', 'VM', 'T]
    def load_data(self, axis_selection_tuple, full_data_norm, downsample_flag, downsample_factor,
                  milling_type='HM', speed=50, window_size=50):
        arr_list = []
        label_list = []
        # print('here')
        if speed == 50:
            # print('here')
            milling_file_path = os.path.join(self.BASE_DIR, milling_type)
            for grit_folder in os.listdir(milling_file_path):
                # print('here')
                grit_folder_path = os.path.join(milling_file_path, grit_folder)
                for file_name in os.listdir(grit_folder_path):
                    if 'Processed' in file_name and 'Speed100' not in file_name:
                        # print(file_name)
                        # print(file_name)
                        granularity_number = file_name.split('_')[0]
                        # print(granularity_number)
                        # if granularity_number in ['HM2', 'VM2', 'T2', 'HM4', 'VM4', 'T4um']:
                        file_path = os.path.join(grit_folder_path, file_name)
                        df = pd.read_csv(file_path)
                        data_arr = df.to_numpy()
                        n_samples, _ = data_arr.shape
                        data_arr = np.round(self.interpolate(data_arr,
                                                    interp_to_frame_len=int(n_samples/downsample_factor),
                                                    method='cubic'), 5)
                        # data_arr_T = self.transform_axes(data_arr)
                        data_arr_T = data_arr
                        if downsample_flag:
                            data_arr_T = data_arr_T[::6, :]
                        if full_data_norm:
                            data_arr_T = self.scaler.fit_transform(data_arr_T)
                        # data_arr_T = data_arr
                        split_window_arr, window_split_label_list = self.split_sweep_into_windows(data_arr_T,
                                                                                                  window_size=window_size,
                                                                                                  label=granularity_number,
                                                                                                  axis_selection_tuple=axis_selection_tuple)
                        arr_list.append(split_window_arr)
                        label_list = label_list + window_split_label_list
        elif speed==100:
            milling_file_path = os.path.join(self.BASE_DIR, milling_type)
            for grit_folder in os.listdir(milling_file_path):
                grit_folder_path = os.path.join(milling_file_path, grit_folder)
                for file_name in os.listdir(grit_folder_path):
                    if 'Processed' in file_name and 'Speed100' in file_name:
                        # print(file_name)
                        granularity_number = file_name.split('_')[0]
                        # print(granularity_number)
                        # if granularity_number in ['HM2', 'VM2', 'T2', 'HM4', 'VM4', 'T4um']:
                        file_path = os.path.join(grit_folder_path, file_name)
                        df = pd.read_csv(file_path)
                        data_arr = df.to_numpy()
                        data_arr_T = self.transform_axes(data_arr)
                        if downsample_flag:
                            data_arr_T = data_arr_T[::6, :]
                        if full_data_norm:
                            data_arr_T = self.scaler.fit_transform(data_arr_T)
                        split_window_arr, window_split_label_list = self.split_sweep_into_windows(data_arr_T,
                                                                                                  window_size=window_size,
                                                                                                  label=granularity_number,
                                                                                                  axis_selection_tuple=axis_selection_tuple)
                        arr_list.append(split_window_arr)
                        label_list = label_list + window_split_label_list
        return np.concatenate(arr_list), np.vstack(label_list)

    # ['HM', 'VM', 'T]
    def load_full_data_with_labels(self, speed=50, window_size=50, axis_selection_tuple=(0, 1, 2, 3, 4, 5),
                                   full_data_norm=False, downsample_flag=False, downsample_factor=1):
        full_arr_list = []
        full_label_list = []
        for surface_type in ['HM', 'T', 'VM']:
            arr, label_arr = self.load_data(milling_type=surface_type, speed=speed, window_size=window_size,
                                            axis_selection_tuple=axis_selection_tuple,
                                            full_data_norm=full_data_norm, downsample_flag=downsample_flag,
                                            downsample_factor=downsample_factor)
            full_arr_list.append(arr)
            full_label_list.append(label_arr)
        return np.concatenate(full_arr_list), pd.factorize(np.vstack(full_label_list)[:, 0])[0], pd.factorize(np.vstack(full_label_list)[:, 0])[1], np.unique(np.vstack(full_label_list)[:, 0])

d = DataLoaderPressureSensor()
X, Y, classes, d = d.load_full_data_with_labels(downsample_factor=1/0.999)

print(X.shape)
print(Y.shape)
# print(X[0][:, 3:])

print(classes)
print(d)
np.save('texture_label', Y)
np.save('texture_data', X)

