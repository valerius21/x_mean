from nuscenes.utils.data_classes import RadarPointCloud

from data_fusion.definitions import nusc, DATASET

"""
# Exercise 1
"""


def get_pcd_data(nusc_filepath: str):
    radar_point_cloud = RadarPointCloud.from_file(nusc_filepath)
    points = radar_point_cloud.points
    x = points[0]
    y = points[1]
    vx_comp = points[8]
    vy_comp = points[9]

    return {
        'file': nusc_filepath,
        'x': x,
        'y': y,
        'vx_comp': vx_comp,
        'vy_comp': vy_comp,
        'v_comp': (vx_comp ** 2 + vy_comp ** 2) ** 0.5,
        'radar_point_cloud': radar_point_cloud
    }


def extract_channel_from_file(channel: str):
    filename = nusc.get('sample_data', channel)['filename']
    filename = f'{DATASET}/{filename}'
    return get_pcd_data(filename)


def extract_samples_from_scene(scene: dict):
    """extract all the samples related to the given scene."""
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    samples = list()

    fst = nusc.get('sample', first_sample_token)
    next_token = fst['next']
    while True:
        current = nusc.get('sample', next_token)
        samples.append(current)
        next_token = current['next']
        if next_token == last_sample_token:
            return samples


def convert_binary_data_to_coordinates_and_velocity(sample: dict):
    data = sample['data']
    return {
        'RADAR_FRONT': extract_channel_from_file(data['RADAR_FRONT']),
        'RADAR_FRONT_LEFT': extract_channel_from_file(data['RADAR_FRONT_LEFT']),
        'RADAR_FRONT_RIGHT': extract_channel_from_file(data['RADAR_FRONT_RIGHT']),
        'RADAR_BACK_LEFT': extract_channel_from_file(data['RADAR_BACK_LEFT']),
        'RADAR_BACK_RIGHT': extract_channel_from_file(data['RADAR_BACK_RIGHT']),
        'data': data
    }
