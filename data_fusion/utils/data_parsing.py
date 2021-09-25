from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

import numpy as np

from data_fusion.definitions import VERSION, DATASET
from data_fusion.utils.binary_files import convert_binary_data_to_coordinates_and_velocity, extract_samples_from_scene

nusc = NuScenes(version=VERSION, dataroot=DATASET, verbose=True)
sc = nusc.scene[0]

samples_from_scene = extract_samples_from_scene(sc)


def load_files():
    """makes a biggo matrix containing all infos about 2 a)."""
    channels = [
        'RADAR_FRONT',
        'RADAR_FRONT_LEFT',
        'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT',
        'RADAR_BACK_RIGHT',
    ]
    samples = [convert_binary_data_to_coordinates_and_velocity(sc) for sc in samples_from_scene]
    scene_points = list()
    for sample in samples:
        x, y, z, vx_comp, vy_comp, pointclouds = list(), list(), list(), list(), list(), list()
        ego_pose_coords = []
        for channel in channels:
            pc = sample[channel]['radar_point_cloud']
            radar_token = sample['data'][channel]
            current_radar = nusc.get('sample_data', radar_token)
            ego_pose = nusc.get('ego_pose', current_radar['ego_pose_token'])
            calibrated_sensor = nusc.get('calibrated_sensor', current_radar['calibrated_sensor_token'])
            sensor_to_car = transform_matrix(calibrated_sensor['translation'],
                                             Quaternion(calibrated_sensor['rotation'], inverse=False))
            car_to_world = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation'], inverse=False))

            sensor_to_world = np.dot(car_to_world, sensor_to_car)

            pc.transform(sensor_to_world)

            pointclouds.append(pc)

            ego_pose_coords = ego_pose['translation']

            # combine radar

            for i in range(pc.points.shape[1]):
                x.append(pc.points[0][i])
                y.append(pc.points[1][i])
                z.append(pc.points[2][i])  # redundant?
                vx_comp.append(pc.points[8][i])  #
                vy_comp.append(pc.points[9][i])
        scene_points.append([
            np.asarray(x),
            np.asarray(y),
            np.asarray(z),
            np.asarray(vx_comp),
            np.asarray(vy_comp),
            np.asarray(pointclouds),
            np.asarray(ego_pose_coords)
        ])

    return np.asarray(scene_points, dtype=object)


result = load_files()
scene_anns = [s['anns'] for s in samples_from_scene]
