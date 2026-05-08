import os
import json
import numpy as np
import constants as co

JSON_DATA_DIR = co.JSON_DATA_DIR

def localToGlobalCoordinateConversion(measurements_tree):
    geometry_data = loadSurfaceGeometryData()
    surface_lookup = createSurfaceLookup(geometry_data)
    fields = ["volume_id", "layer_id", "surface_id", "extra_id", "rec_loc0", "rec_loc1"]
    events = measurements_tree.arrays(fields, library="np")
    global_coords = matchCoordinates(events, surface_lookup)
    return global_coords

def createSurfaceLookup(geometry_data):
    surface_lookup = {}
    for entry in geometry_data:
        key = (
            entry["volume"],
            entry["layer"],
            entry["sensitive"]
        )
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

def matchCoordinates(events, surface_lookup):
    global_coords = []
    for i in range(len(events["rec_loc0"])):
        key = (
            int(events["volume_id"][i]),
            int(events["layer_id"][i]),
            int(events["surface_id"][i])
        )
        if key in surface_lookup:
            transform = surface_lookup[key]
            loc0 = events["rec_loc0"][i]
            loc1 = events["rec_loc1"][i]
            global_coord = localToGlobalCalculation(loc0, loc1, transform)
            global_coords.append(global_coord)
        else:
            print(f"Warning: No transform found for key {key}")
            global_coords.append([np.nan, np.nan, np.nan])
    global_coords = np.array(global_coords)
    return global_coords