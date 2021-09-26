from nuscenes.nuscenes import NuScenes

# basic config and variables


VERSION = 'v1.0-mini'
DATASET = '/home/valerius/data/sets/nuscenes'
SCENE_LENGTH = 37

Y_LIM_MIN = 1000
Y_LIM_MAX = 1200

X_LIM_MIN = 350
X_LIM_MAX = 450

VEHICLE_LENS = 1.3  # Amplification of the bounding box to get all radar points

nusc = NuScenes(version=VERSION, dataroot=DATASET, verbose=True)

# Extracting Annotation Tokens for the scene
