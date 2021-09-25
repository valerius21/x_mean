from nuscenes.nuscenes import NuScenes

# basic config and variables


VERSION = 'v1.0-mini'
DATASET = '/home/valerius/data/sets/nuscenes'
SCENE_LENGTH = 37

Y_LIM_MIN = 900
Y_LIM_MAX = 1300

X_LIM_MIN = 300
X_LIM_MAX = 550

VEHICLE_LENS = 5.3 # Amplification of the bounding box to get all radar points

nusc = NuScenes(version=VERSION, dataroot=DATASET, verbose=True)

# Extracting Annotation Tokens for the scene
