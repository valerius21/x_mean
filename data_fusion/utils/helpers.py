from statistics import mean

import numpy as np

from data_fusion.definitions import SCENE_LENGTH
from data_fusion.utils.data import base_data


class NoRowException(Exception):
    """
    Raise, when there is no fitting data for a current timestamp.
    """
    pass


def get_point_coords_from_v_comp(vx_comp, vy_comp, ego_x_coord, ego_y_coord, vehicle_x, vehicle_y):
    delta_x = vehicle_x - ego_x_coord
    delta_y = vehicle_y - ego_y_coord
    length_total = np.sqrt(delta_x ** 2 + delta_y ** 2)
    length_should = np.sqrt(vx_comp ** 2 + vy_comp ** 2)
    scalar = length_total / length_should
    return ego_x_coord, ego_y_coord, delta_x / scalar, delta_y / scalar


def reduce_measurement(row_index: int, base_data):
    """
    average all measurements and return it with ground truth
    """
    row = [e for e in base_data if e['row'] == row_index]

    if not row:
        # raise Exception("No rows for this index!")
        return {}
    try:
        res = {
            'x': mean([e['x'] for e in row]),
            'y': mean([e['y'] for e in row]),
            'vx': mean([e['vx'] for e in row]),
            'vy': mean([e['vy'] for e in row]),
            'gt_x': row[0]['gt_x'],
            'gt_y': row[0]['gt_y']
        }
    except ZeroDivisionError as ze:
        print('[reduce_measurement] ZeroDivision')
        raise ze

    return res


def get_missing():
    s = [r['row'] for r in base_data]
    s = set(s)
    ni = [i for i in range(37) if i not in s]
    return ni
