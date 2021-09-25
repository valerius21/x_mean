# Helper Class
from data_fusion.utils.data_parsing import nusc


class Vehicle:
    """
    Holding an annotated Vehicle
    """

    token = "None"
    x = None
    y = None
    w = None
    h = None
    angle = 0.0

    def __init__(self, annotation: dict):
        self.token = annotation['token']
        sample_translation = annotation['translation']
        self.x = sample_translation[0]
        self.y = sample_translation[1]
        sample_size = annotation['size']
        self.h = sample_size[1]
        self.w = sample_size[0]

    def get_left_corner(self):
        x_left = self.x - (self.w / 2)
        y_left = self.y - (self.h / 2)
        return x_left, y_left

    def get_trans(self):
        return self.x, self.y

    def __repr__(self):
        return f'TOKEN:{self.token}\tTRANSL:{self.get_trans()}\tLEFT_CORNER:{self.get_left_corner()}'


def get_vehicles_from_sample(ann_ids: [str]):
    """
    Extract vehicles from given sample with annotation ID.
    """
    anns = [nusc.get('sample_annotation', a_id) for a_id in ann_ids]
    anns = [v for v in anns if 'vehicle.' in v['category_name']]
    veh = [Vehicle(a) for a in anns]
    return veh
