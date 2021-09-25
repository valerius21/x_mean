import numpy as np

from data_fusion.definitions import SCENE_LENGTH
from data_fusion.entities.annotations import FOCUS_VEHICLES
from data_fusion.utils.data_parsing import result

avg_x, avg_y = [], []
_result_iterator = iter(result)


def get_x_y_vx_vy_row_gt(i: int):
    collector = []
    try:
        row = next(_result_iterator)
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

    lens = 5.3
    current_x_min = veh_coords_x[i]
    current_y_min = veh_coords_y[i]
    width = vehicles[i].w * lens
    height = vehicles[i].h * lens
    coords, new_x, new_y = [], [], []
    for j, p in enumerate(list(zip(x, y))):
        is_in_x = current_x_min <= p[0] <= current_x_min + width
        is_in_y = current_y_min <= p[1] <= current_y_min + height
        if is_in_x and is_in_y:
            coords.append(p)
            current_data = {
                'x': x[j],
                'y': y[j],
                'vx': vx[j],
                'vy': vy[j],
                'row': i,
                'gt_x': veh_coords_x[i],
                'gt_y': veh_coords_y[i]
            }
            collector.append(current_data)

    new_x = ([t[0] for t in coords])
    new_y = ([t[1] for t in coords])

    new_x = sum(new_x) / len(new_x)
    new_y = sum(new_y) / len(new_y)

    avg_x.append(new_x)
    avg_y.append(new_y)

    return collector


base_data = [get_x_y_vx_vy_row_gt(i) for i in range(SCENE_LENGTH)][0]
