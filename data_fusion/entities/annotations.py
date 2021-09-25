from data_fusion.entities.Vehicle import Vehicle
from data_fusion.utils.data_parsing import nusc, sc

"""
# Exercise 3 A
"""


def extract_sample_annotations_from_scene(scene: dict):
    """
    extract all the sample annotations related to the given scene.
    """
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']

    sample_annotations = []

    fst = nusc.get('sample', first_sample_token)
    current = fst

    while True:
        [sample_annotations.append(a) for a in nusc.sample_annotation
         if a['sample_token'] == current['token'] and '' in a['category_name']]
        if current['token'] == last_sample_token:
            return sample_annotations
        current = nusc.get('sample', current['next'])


def walk_annotations(d, collector=None):
    """
    recursivly get all annotations from sample.
    """
    if collector is None:
        collector = []
    if d['next'] == '':
        return collector
    collector.append(d)
    return walk_annotations(nusc.get('sample_annotation', d['next']), collector)


def _focus_the_vehicles():
    # Temporary collector
    data = dict()

    for ann in extract_sample_annotations_from_scene(sc):
        if ann['prev'] == '' and 'c3246a1e22a14fcb878aa61e69ae3329' in ann['attribute_tokens']:
            # and nusc.get('attribute', ann['attribute_tokens'][0])['name'] == 'vehicle.moving':
            data[ann['token']] = len(walk_annotations(ann))

    # Vehicle to follow
    last = nusc.get('sample_annotation', 'd02856dbe859476f9635f449d23aa211')

    # Collector for annotations of selected vehicle
    FOCUS = []

    # extracting all annotation of that vehicle
    while last['prev'] != '':
        FOCUS.append(last)
        last = nusc.get('sample_annotation', last['prev'])
    return [Vehicle(nusc.get('sample_annotation', f['token'])) for f in FOCUS]


# Fabricate @Vehicle objects for the selected vehicle
FOCUS_VEHICLES = _focus_the_vehicles()
