import pickle

import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tools.io_f import read_data_file, get_path_info
from tools.compute_f import split_ts_seq, compute_steps, compute_headings, \
    compute_stride_length, compute_step_heading, compute_rel_positions, correct_positions

floor_map = {"B3": -3, "B2": -2, "B1": -1, "F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4, "F6": 5, "F7": 6, "F8": 7,
             "F9": 8, "F10": 9, "F11": 10, "F12": 11,
             "3B": -3, "2B": -2, "1B": -1, "1F": 0, "2F": 1, "3F": 2, "4F": 3, "5F": 4, "6F": 5, "7F": 6, "8F": 7,
             "9F": 8, "10F": 9, "11F": 10, "12F": 11,
             "LG3": -3, "LG2": -2, "LG1": -1, "L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4, "L6": 5, "L7": 6, "L8": 7,
             "L9": 8, "L10": 9, "L11": 10, "L12": 11,
             "B": -2, "G": -1, "LM": 1, "P1": -3, "P2": -4, "BM": -2, "BF": -1}


def calibrate_magnetic_wifi_ibeacon_to_step(path_file_list, data_from='train'):
    mwi_data = dict()
    for path_filename in path_file_list:
        # print(f'Processing {path_filename}...')
        path_data = read_data_file(path_filename)
        accelerometer_data = path_data.acce
        magnetic_data = path_data.magn
        rotation_data = path_data.ahrs
        wifi_data = path_data.wifi
        ibeacon_data = path_data.ibeacon
        pos_data = path_data.waypoint

        step_timestamps, step_indexs, step_acce_max_mins = compute_steps(accelerometer_data)
        headings = compute_headings(rotation_data)
        stride_lengths = compute_stride_length(step_acce_max_mins)  # [timestamp, stride_length]
        step_headings = compute_step_heading(step_timestamps, headings)  # [timestamp, heading]
        relative_pos = compute_rel_positions(stride_lengths, step_headings)  # [timestamp, dx, dy]

        sep_tss = np.unique(magnetic_data[:, 0].astype(float))
        mag_data_list = split_ts_seq(magnetic_data, sep_tss)

        if data_from == 'train':
            floor = floor_map[path_filename.parent.stem]
            step_positions = correct_positions(relative_pos, pos_data)
            for magn_ds in mag_data_list:
                diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
                if len(diff) != 0:
                    index = np.argmin(diff)
                    target_fxy_key = tuple(np.append(step_positions[index, 1:3], floor))
                    if target_fxy_key in mwi_data:
                        mwi_data[target_fxy_key]['magnetic'] = np.append(mwi_data[target_fxy_key]['magnetic'], magn_ds,
                                                                         axis=0)
                    else:
                        mwi_data[target_fxy_key] = {'magnetic': magn_ds, 'wifi': np.zeros((0, 5)),
                                                    'ibeacon': np.zeros((0, 3))}
            if wifi_data.size != 0:
                sep_tss = np.unique(wifi_data[:, 0].astype(float))
                wifi_data_list = split_ts_seq(wifi_data, sep_tss)
                for wifi_ds in wifi_data_list:
                    diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                    if len(diff) != 0:
                        index = np.argmin(diff)
                        target_fxy_key = tuple(np.append(step_positions[index, 1:3], floor))
                        if target_fxy_key in mwi_data:
                            mwi_data[target_fxy_key]['wifi'] = np.append(mwi_data[target_fxy_key]['wifi'], wifi_ds,
                                                                         axis=0)
                        else:
                            mwi_data[target_fxy_key] = {'magnetic': np.zeros((0, 4)), 'wifi': wifi_ds,
                                                        'ibeacon': np.zeros((0, 3))}
            if ibeacon_data.size != 0:
                sep_tss = np.unique(ibeacon_data[:, 0].astype(float))
                ibeacon_data_list = split_ts_seq(ibeacon_data, sep_tss)
                for ibeacon_ds in ibeacon_data_list:
                    diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
                    if len(diff) != 0:
                        index = np.argmin(diff)
                        target_fxy_key = tuple(np.append(step_positions[index, 1:3], floor))
                        if target_fxy_key in mwi_data:
                            mwi_data[target_fxy_key]['ibeacon'] = np.append(mwi_data[target_fxy_key]['ibeacon'],
                                                                            ibeacon_ds, axis=0)
                        else:
                            mwi_data[target_fxy_key] = {'magnetic': np.zeros((0, 4)), 'wifi': np.zeros((0, 5)),
                                                        'ibeacon': ibeacon_ds}
        elif data_from == 'test':
            for mag_ds in mag_data_list:
                diff = np.abs(step_timestamps - float(mag_ds[0, 0]))
                if len(diff) != 0:
                    index = np.argmin(diff)
                    timestamp_key = step_timestamps[index]
                    if timestamp_key in mwi_data:
                        mwi_data[timestamp_key]['magnetic'] = np.append(mwi_data[timestamp_key]['magnetic'], mag_ds,
                                                                        axis=0)
                    else:
                        mwi_data[timestamp_key] = {'magnetic': mag_ds, 'wifi': np.zeros((0, 5)),
                                                   'ibeacon': np.zeros((0, 3))}
            if wifi_data.size != 0:
                sep_tss = np.unique(wifi_data[:, 0].astype(float))
                wifi_data_list = split_ts_seq(wifi_data, sep_tss)
                for wifi_ds in wifi_data_list:
                    diff = np.abs(step_timestamps - float(wifi_ds[0, 0]))
                    if len(diff) != 0:
                        index = np.argmin(diff)
                        timestamp_key = step_timestamps[index]
                        if timestamp_key in mwi_data:
                            mwi_data[timestamp_key]['wifi'] = np.append(mwi_data[timestamp_key]['wifi'], wifi_ds,
                                                                        axis=0)
                        else:
                            mwi_data[timestamp_key] = {'magnetic': np.zeros((0, 4)), 'wifi': wifi_ds,
                                                       'ibeacon': np.zeros((0, 3))}
            if ibeacon_data.size != 0:
                sep_tss = np.unique(ibeacon_data[:, 0].astype(float))
                ibeacon_data_list = split_ts_seq(ibeacon_data, sep_tss)
                for ibeacon_ds in ibeacon_data_list:
                    diff = np.abs(step_timestamps - float(ibeacon_ds[0, 0]))
                    if len(diff) != 0:
                        index = np.argmin(diff)
                        timestamp_key = step_timestamps[index]
                        if timestamp_key in mwi_data:
                            mwi_data[timestamp_key]['ibeacon'] = np.append(mwi_data[timestamp_key]['ibeacon'],
                                                                           ibeacon_ds, axis=0)
                        else:
                            mwi_data[timestamp_key] = {'magnetic': np.zeros((0, 4)), 'wifi': np.zeros((0, 5)),
                                                       'ibeacon': ibeacon_ds}
    return mwi_data


def extract_magnetic_strength(mwi_data):
    magnetic_strength = {}
    for position_key in mwi_data:
        # print(f'Position: {position_key}')

        magnetic_data = mwi_data[position_key]['magnetic']
        magnetic_s = np.mean(np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
        magnetic_strength[position_key] = magnetic_s

    return magnetic_strength


def extract_wifi_rssi(mwi_data):
    wifi_rssi = {}
    for position_key in mwi_data:
        # print(f'Position: {position_key}')

        wifi_data = mwi_data[position_key]['wifi']
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])

            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = dict()
                position_rssi[position_key] = np.array([rssi, 1])

            wifi_rssi[bssid] = position_rssi

    return wifi_rssi


def extract_ibeacon_rssi(mwi_data):
    ibeacon_rssi = {}
    for position_key in mwi_data:
        # print(f'Position: {position_key}')

        ibeacon_data = mwi_data[position_key]['ibeacon']
        for ibeacon_d in ibeacon_data:
            ummid = ibeacon_d[1]
            rssi = int(ibeacon_d[2])

            if ummid in ibeacon_rssi:
                position_rssi = ibeacon_rssi[ummid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            ibeacon_rssi[ummid] = position_rssi

    return ibeacon_rssi


def extract_wifi_count(mwi_data):
    wifi_counts = {}
    for position_key in mwi_data:
        # print(f'Position: {position_key}')

        wifi_data = mwi_data[position_key]['wifi']
        count = np.unique(wifi_data[:, 2]).shape[0]
        wifi_counts[position_key] = count

    return wifi_counts


def calibrate_train_raw_to_step(train_site_id_list):
    for cnt, site_id in enumerate(train_site_id_list):
        print(f'{cnt + 1}/{len(train_site_id_list)} - Start processing site: {site_id} ...')
        folders = [f for f in Path(f'../data/train/{site_id}').resolve().iterdir() if f.is_dir()]
        all_file_list = list()
        for folder in folders:
            file_list = [t for t in folder.iterdir() if t.glob('*.txt')]
            all_file_list.extend(file_list)
        site_mwi = calibrate_magnetic_wifi_ibeacon_to_step(all_file_list, data_from='train')
        with open(f'../data/pickle_file/train/{site_id}_mwi_data.pickle', 'wb') as f:
            pickle.dump(site_mwi, f)


def calibrate_test_raw_to_step(test_path_dir):
    test_path_list = list(Path(test_path_dir).resolve().glob('*.txt'))
    for cnt, test_path in enumerate(test_path_list):
        info = get_path_info(test_path, 'test')
        site_id = info['SiteID']
        print(f"{cnt + 1}/{len(test_path_list)} - Start processing path {info['SiteName']}_{site_id}_{test_path.stem}...")
        path_mwi = calibrate_magnetic_wifi_ibeacon_to_step([test_path], data_from='test')
        with open(f'../data/pickle_file/test/{site_id}_{test_path.stem}_mwi_test_data.pickle', 'wb') as f:
            pickle.dump(path_mwi, f)


# if __name__ == '__main__':
#     siteId = '5cd56bc2e2acfd2d33b640cc'
#     path_file = '2F/5d0b396a7a560f00088a9407.txt'
#     test_dir = Path('../data/train')
#     test_dir = test_dir / siteId / path_file
#     # step_analysis(test_dir)
#
#     # mwi = calibrate_magnetic_wifi_ibeacon_to_step([test_dir], 'train')
#     # print(mwi.keys())
#
#     # calibrate_train_raw_to_step([siteId])
#     calibrate_test_raw_to_step('../data/test')
#
#     # # wifi_rssi = extract_wifi_rssi(mwi)
#     # # pd.DataFrame(wifi_rssi).to_csv('wifi_rssi.csv')
#     #
#     # # ibeacon_data = extract_ibeacon_rssi(mwi)
#     # # pd.DataFrame(ibeacon_data).to_csv('ibeacon.csv')
#     # # ibeacon_df = pd.DataFrame(ibeacon_data)
#     # # print(ibeacon_df.size)
#     #
#     # # pd.DataFrame(magnetic, index=[0]).to_csv('mag.csv')
#     # magnetic = pd.DataFrame(extract_magnetic_strength(mwi), index=[0])
#     # points = magnetic.columns.to_numpy()
#     # m = magnetic.values[0].tolist()
#     # print(m)
#     # x = list()
#     # y = list()
#     # for p in points:
#     #     x.append(p[0])
#     #     y.append(p[1])
#     # mag = pd.DataFrame(np.stack((x, y, m))).T
#     # mag.columns = ['x', 'y', 'mag']
#     # print(mag)
#     # # mag = mag.pivot('x', 'y', 'mag')
#     # # sns.heatmap(mag)
#     # sns.lineplot(x=mag.index, y='mag', data=mag)
#     # plt.show()
