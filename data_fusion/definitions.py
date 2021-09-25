from nuscenes.nuscenes import NuScenes

# basic config and variables


VERSION = 'v1.0-mini'
DATAROOT = '/home/valerius/data/sets/nuscenes'
SCENE_LENGTH = 37

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

# Extracting Annotation Tokens for the scene
