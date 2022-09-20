import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataLoaderLaser:
    def __init__(self, BASE_DIR='./opticalNCDT'):
        self.BASE_DIR = BASE_DIR
        self.scaler = StandardScaler()

    def split_sweep_into_windows(self, arr, window_size, label, axis_selection_tuple):
        n_samples, n_dim = arr.shape
        window_arr = np.zeros((n_samples // window_size, window_size, len(axis_selection_tuple)))
        window_label_list = [label] * (n_samples // window_size)
        for start_index in np.arange(0, (n_samples // window_size) * window_size, window_size):
            window_arr[start_index // window_size] = arr[start_index:start_index + window_size][:, axis_selection_tuple]
        return window_arr, window_label_list

    # three milling types ['HM', 'VM', 'T]
    def load_data(self, axis_selection_tuple, full_data_norm, milling_type='HM', speed=50, window_size=50):
        arr_list = []
        label_list = []
        if speed == 50:
            milling_file_path = os.path.join(self.BASE_DIR, 'speed50', milling_type)
            for grit_folder in os.listdir(milling_file_path):
                grit_folder_path = os.path.join(milling_file_path, grit_folder)
                for file_name in os.listdir(grit_folder_path):
                    # print(file_name)
                    granularity_number = file_name.split('_')[0]
                    # print(granularity_number)
                    file_path = os.path.join(grit_folder_path, file_name)
                    df = pd.read_csv(file_path)
                    data_arr = df.to_numpy()
                    # print(data_arr.shape)
                    if full_data_norm:
                        data_arr = self.scaler.fit_transform(data_arr)
                    # data_arr_T = data_arr
                    for i in range(3):
                        split_window_arr, window_split_label_list = self.split_sweep_into_windows(data_arr[i::3],
                                                                                                  window_size=window_size,
                                                                                                  label=granularity_number,
                                                                                                  axis_selection_tuple=axis_selection_tuple)
                        arr_list.append(split_window_arr)
                        label_list = label_list + window_split_label_list
        elif speed == 100:
            milling_file_path = os.path.join(self.BASE_DIR, 'speed100', milling_type)
            for grit_folder in os.listdir(milling_file_path):
                grit_folder_path = os.path.join(milling_file_path, grit_folder)
                for file_name in os.listdir(grit_folder_path):
                    # print(file_name)
                    granularity_number = file_name.split('_')[0]
                    # print(granularity_number)
                    file_path = os.path.join(grit_folder_path, file_name)
                    df = pd.read_csv(file_path)
                    data_arr = df.to_numpy()
                    # print(data_arr.shape)
                    if full_data_norm:
                        data_arr = self.scaler.fit_transform(data_arr)
                    # data_arr_T = data_arr
                    for i in range(3):
                        split_window_arr, window_split_label_list = self.split_sweep_into_windows(data_arr[i::3],
                                                                                                  window_size=window_size,
                                                                                                  label=granularity_number,
                                                                                                  axis_selection_tuple=axis_selection_tuple)
                        arr_list.append(split_window_arr)
                        label_list = label_list + window_split_label_list
        return np.concatenate(arr_list), np.vstack(label_list)

    # ['HM', 'VM', 'T]
    def load_full_data_with_labels(self, speed=50, window_size=50, axis_selection_tuple=(0, ),
                                   full_data_norm=False, downsample_flag=False):
        full_arr_list = []
        full_label_list = []
        for surface_type in ['HM', 'VM', 'T']:
            arr, label_arr = self.load_data(milling_type=surface_type, speed=speed, window_size=window_size,
                                            axis_selection_tuple=axis_selection_tuple,
                                            full_data_norm=full_data_norm)
            full_arr_list.append(arr)
            full_label_list.append(label_arr)
        return np.concatenate(full_arr_list), pd.factorize(np.vstack(full_label_list)[:, 0])[0], \
               pd.factorize(np.vstack(full_label_list)[:, 0])[1], np.unique(np.vstack(full_label_list)[:, 0])

# d = DataLoaderLaser()
# X, Y, classes, class_name = d.load_full_data_with_labels(speed=100)
# print(class_name)
