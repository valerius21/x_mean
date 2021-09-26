import itertools

import numpy as np

from data_fusion.definitions import SCENE_LENGTH, VEHICLE_LENS
from data_fusion.entities.annotations import FOCUS_VEHICLES
from data_fusion.utils.data_parsing import result

avg_x, avg_y = [], []
_result_iterator = iter(result)


def _append_new_avg_coordinates(new_x, new_y):
    len_x, len_y = len(new_x), len(new_y)
    if len_x == 0 or len_y == 0:
        return
    ax = sum(new_x) / len(new_x)
    ay = sum(new_y) / len(new_y)
    avg_x.append(ax)
    avg_y.append(ay)


def get_x_y_vx_vy_row_gt(count: int):
    collector = []
    try:
        row = result[count]
    except StopIteration:
        return
    x, y = row[0], row[1]
    vx, vy = row[3], row[4]

    vehicles = FOCUS_VEHICLES
    veh_coords = [v.get_trans() for v in vehicles]
    veh_coords_x = [x[0] for x in veh_coords]
    veh_coords_y = [x[1] for x in veh_coords]
    veh_coords_x.reverse()
    veh_coords_y.reverse()

    veh_coords_x = np.asarray(veh_coords_x)
    veh_coords_y = np.asarray(veh_coords_y)

    current_x_min = veh_coords_x[count]
    current_y_min = veh_coords_y[count]
    width = vehicles[count].w * VEHICLE_LENS
    height = vehicles[count].h * VEHICLE_LENS
    coords, new_x, new_y = [], [], []
    for i, p in enumerate(list(zip(x, y))):
        is_in_x = current_x_min <= p[0] <= current_x_min + width
        is_in_y = current_y_min <= p[1] <= current_y_min + height
        if is_in_x and is_in_y:
            coords.append(p)
            current_data = {
                'x': x[i],
                'y': y[i],
                'vx': vx[i],
                'vy': vy[i],
                'v': np.sqrt(vx[i] ** 2 + vy[i] ** 2),
                'alpha': np.arccos(vx[i] / np.sqrt(vx[i] ** 2 + vy[i] ** 2)),
                'row': count,
                'gt_x': veh_coords_x[count],
                'gt_y': veh_coords_y[count]
            }
            collector.append(current_data)

    new_x = ([t[0] for t in coords])
    new_y = ([t[1] for t in coords])
    _append_new_avg_coordinates(new_x, new_y)
    return collector


_bd = [get_x_y_vx_vy_row_gt(i) for i in range(SCENE_LENGTH)]
base_data = list(itertools.chain(*_bd))

if __name__ == '__main__':
    current_samples = [r['row'] for r in list(itertools.chain(*_bd))]

