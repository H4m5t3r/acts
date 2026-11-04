import os
import json
from functools import lru_cache
import numpy as np
import constants as co

JSON_DATA_DIR = co.JSON_DATA_DIR

MEASUREMENT_GEOMETRY_FIELDS = [
    "volume_id",
    "layer_id",
    "surface_id",
    "rec_loc0",
    "rec_loc1",
]


def localToGlobalCoordinateConversion(measurements_tree):
    events = measurements_tree.arrays(MEASUREMENT_GEOMETRY_FIELDS, library="np")
    return localToGlobalFromArrays(
        events["volume_id"],
        events["layer_id"],
        events["surface_id"],
        events["rec_loc0"],
        events["rec_loc1"],
    )


def localToGlobalFromArrays(volume_id, layer_id, surface_id, loc0, loc1):
    """Vectorised local -> global conversion of measurement positions. Returns an
    (n, 3) array; rows on surfaces missing from the geometry file are NaN."""
    key_to_index, rotations, translations = loadSurfaceTransforms()
    hit_keys = np.column_stack([volume_id, layer_id, surface_id]).astype(np.int64)
    if len(hit_keys) == 0:
        return np.empty((0, 3))
    # Only the (few thousand) distinct surfaces need a Python-level dict lookup
    unique_keys, inverse = np.unique(hit_keys, axis=0, return_inverse=True)
    unique_index = np.array(
        [key_to_index.get(tuple(int(v) for v in key), -1) for key in unique_keys]
    )
    for key in unique_keys[unique_index < 0]:
        print(f"Warning: No transform found for key {tuple(int(v) for v in key)}")
    surface_index = unique_index[inverse.ravel()]
    valid = surface_index >= 0

    global_coords = np.full((len(hit_keys), 3), np.nan)
    rot = rotations[surface_index[valid]]
    # local position is (loc0, loc1, 0), so only the first two rotation columns enter
    global_coords[valid] = (
        rot[:, :, 0] * np.asarray(loc0, dtype=np.float64)[valid, None]
        + rot[:, :, 1] * np.asarray(loc1, dtype=np.float64)[valid, None]
        + translations[surface_index[valid]]
    )
    return global_coords


@lru_cache(maxsize=None)
def loadSurfaceTransforms(json_data_dir=JSON_DATA_DIR):
    """Parses the detector geometry once per process and returns
    ({(volume, layer, sensitive): index}, rotations (S, 3, 3), translations (S, 3))."""
    surface_lookup = createSurfaceLookup(loadSurfaceGeometryData(json_data_dir))
    key_to_index = {key: i for i, key in enumerate(surface_lookup)}
    rotations = np.stack(
        [
            (
                np.eye(3)
                if transform["rotation"] is None
                else np.array(transform["rotation"], dtype=np.float64).reshape(3, 3)
            )
            for transform in surface_lookup.values()
        ]
    )
    translations = np.stack(
        [
            np.array(transform["translation"], dtype=np.float64)
            for transform in surface_lookup.values()
        ]
    )
    return key_to_index, rotations, translations


def createSurfaceLookup(geometry_data):
    surface_lookup = {}
    for entry in geometry_data:
        key = (entry["volume"], entry["layer"], entry["sensitive"])
        surface_lookup[key] = entry["value"]["transform"]
    return surface_lookup


def loadSurfaceGeometryData(json_data_dir=JSON_DATA_DIR):
    with open(os.path.join(json_data_dir, "event000000000-detector.json"), "r") as f:
        geometry_data = json.load(f)["entries"]
    return geometry_data


def localToGlobalCalculation(loc0, loc1, transform):
    if transform["rotation"] is None:
        rotation = np.eye(3)
    else:
        rotation = np.array(transform["rotation"]).reshape(3, 3)
    translation = np.array(transform["translation"])
    local_vec = np.array([loc0, loc1, 0.0])
    result = rotation @ local_vec + translation
    return result
