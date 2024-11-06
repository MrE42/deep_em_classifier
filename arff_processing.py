import numpy as np
import pandas as pd


def smooth_data(data, col, method="moving_average", window=5):
    return data[col].rolling(window=window, min_periods=1).mean()


def get_velocity(data, attributes, window_width):
    c_min_conf = 0.75
    step = int(np.ceil(window_width / 2))

    speed = np.zeros(data.shape[0])
    direction = np.zeros(data.shape[0])

    time_ind = attributes['time']
    x_ind = attributes['x']
    y_ind = attributes['y']
    conf_ind = attributes['confidence']

    for i in range(data.shape[0]):
        if data.iloc[i, conf_ind] < c_min_conf:
            continue

        start_pos = max(0, i - step)
        end_pos = min(data.shape[0] - 1, i + step)

        while start_pos > 0 and data.iloc[start_pos, conf_ind] < c_min_conf:
            start_pos += 1
        while end_pos < data.shape[0] - 1 and data.iloc[end_pos, conf_ind] < c_min_conf:
            end_pos -= 1

        if start_pos == end_pos:
            continue

        ampl = np.sqrt((data.iloc[end_pos, x_ind] - data.iloc[start_pos, x_ind]) ** 2 +
                       (data.iloc[end_pos, y_ind] - data.iloc[start_pos, y_ind]) ** 2)
        time = (data.iloc[end_pos, time_ind] - data.iloc[start_pos, time_ind]) / 1_000_000
        speed[i] = ampl / time if time != 0 else 0
        direction[i] = np.arctan2(data.iloc[end_pos, y_ind] - data.iloc[start_pos, y_ind],
                                  data.iloc[end_pos, x_ind] - data.iloc[start_pos, x_ind])
    return speed, direction


def get_acceleration(data, attributes, att_speed, att_dir, window_width):
    c_min_conf = 0.75
    step = int(np.ceil(window_width / 2))

    acceleration = np.zeros(data.shape[0])

    time_ind = attributes['time']
    conf_ind = attributes['confidence']
    speed_ind = attributes[att_speed]
    dir_ind = attributes[att_dir]

    for i in range(data.shape[0]):
        if data.iloc[i, conf_ind] < c_min_conf:
            continue

        start_pos = max(0, i - step)
        end_pos = min(data.shape[0] - 1, i + step)

        while start_pos > 0 and data.iloc[start_pos, conf_ind] < c_min_conf:
            start_pos += 1
        while end_pos < data.shape[0] - 1 and data.iloc[end_pos, conf_ind] < c_min_conf:
            end_pos -= 1

        if start_pos == end_pos:
            continue

        vel_start_x = data.iloc[start_pos, speed_ind] * np.cos(data.iloc[start_pos, dir_ind])
        vel_start_y = data.iloc[start_pos, speed_ind] * np.sin(data.iloc[start_pos, dir_ind])
        vel_end_x = data.iloc[end_pos, speed_ind] * np.cos(data.iloc[end_pos, dir_ind])
        vel_end_y = data.iloc[end_pos, speed_ind] * np.sin(data.iloc[end_pos, dir_ind])

        delta_t = (data.iloc[end_pos, time_ind] - data.iloc[start_pos, time_ind]) / 1_000_000
        acc_x = (vel_end_x - vel_start_x) / delta_t if delta_t != 0 else 0
        acc_y = (vel_end_y - vel_start_y) / delta_t if delta_t != 0 else 0
        acceleration[i] = np.sqrt(acc_x ** 2 + acc_y ** 2)
    return acceleration


def annotate_data(arff_file, output_file):
    windows_sizes = [1, 2, 4, 8, 16]

    data, metadata, attributes, relation, comments = load_arff(arff_file)
    comments.append("The number after speed, direction denotes the step size used for calculation.")
    comments.append("Acceleration was calculated between adjacent samples of the low-pass filtered velocity.")

    for step in windows_sizes:
        speed, direction = get_velocity(data, attributes, step)

        speed_att_name = f'speed_{step}'
        dir_att_name = f'direction_{step}'

        data[speed_att_name] = speed
        data[dir_att_name] = direction
        attributes[speed_att_name] = len(attributes)
        attributes[dir_att_name] = len(attributes)

        acceleration = get_acceleration(data, attributes, speed_att_name, dir_att_name, 1)
        acc_att_name = f'acceleration_{step}'

        data[acc_att_name] = acceleration
        attributes[acc_att_name] = len(attributes)

    save_arff(output_file, data, metadata, attributes, relation, comments)


def load_arff(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = [line for line in lines if line.startswith('%@METADATA')]
    relation = next((line.split()[-1] for line in lines if line.startswith('@RELATION')), 'gaze_data')
    comments = [line.strip() for line in lines if line.startswith('%')]
    attribute_lines = [line for line in lines if line.startswith('@ATTRIBUTE')]

    columns = [line.split()[1] for line in attribute_lines]
    data_lines = lines[lines.index('@DATA\n') + 1:]
    data = pd.DataFrame([line.strip().split(',') for line in data_lines], columns=columns).apply(pd.to_numeric,
                                                                                                 errors='coerce')

    attributes = {col: i for i, col in enumerate(columns)}
    return data, metadata, attributes, relation, comments


def save_arff(file_path, data, metadata, attributes, relation, comments):
    with open(file_path, 'w') as f:
        for comment in comments:
            f.write(f"{comment}\n")

        f.write(f"@RELATION {relation}\n\n")
        for meta in metadata:
            f.write(f"{meta}\n")

        for attr in attributes:
            f.write(f"@ATTRIBUTE {attr} numeric\n")

        f.write("\n@DATA\n")
        for row in data.itertuples(index=False):
            f.write(",".join(map(str, row)) + "\n")
