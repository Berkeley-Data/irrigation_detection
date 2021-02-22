import argparse
import math
import os
import csv
import json
import numpy as np
import os
from glob import glob

"""
# Sample metadata from BigEarthNet dataset
{"labels": ["Continuous urban fabric", "Discontinuous urban fabric", "Permanently irrigated land", "Rice fields"], 
"coordinates": {"ulx": 540780, "uly": 4312440, "lrx": 541980, "lry": 4311240}, "projection": 
"PROJCS[\"WGS 84 / UTM zone 29N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,
AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],
UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AXIS[\"Latitude\",NORTH],AXIS[\"Longitude\",EAST],
AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",-9],
PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,
AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32629\"]]", 
"tile_source": "S2A_MSIL1C_20171221T112501_N0206_R037_T29SND_20171221T114356.SAFE", "acquisition_date": "2017-12-21 11:25:01"}
"""

# From the above file zone=29, easting=540780, northing=4312440, northernHemisphere=TRUE for BigEarthNet and
# If "uly": 4312440 is positive, then northernHemisphere=TRUE, else northernHemisphere=FALSE
def convert_utm_to_latlng(zone, easting, northing):
    northernHemisphere = True
    if northing < 0:
        northernHemisphere = False

    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996
    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))
    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))
    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0
    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)
    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))
    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0
    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2
    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24
    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720
    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi
    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi
    if not northernHemisphere:
        latitude = -latitude
    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return (latitude, longitude)

# Parse the fMoW dataset metadata files and images, and create a csv file indicating image name location, timestamp, and country
def parse_fmow_dataset(root_folder = "/workspace/app/data/raw/fMoW/fmow-rgb", csv_file="fmow-rgb-all-train-testing.csv"):
    folder_path_list = []
    csv_file_to_parse = os.path.join(root_folder, csv_file)
    if not os.path.exists(csv_file_to_parse):
        print('ERROR: file', csv_file_to_parse, 'does not exist')

    splits = glob(f"{csv_file_to_parse}")
    patch_names_list = []
    for file in splits:
        print(file)
        with open(file, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                patch_names_list.append(row[0].strip())
    print(patch_names_list)

    # writing to csv file
    with open("temp.csv", 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # Parse each json file
        lt_ln_data = {}
        image_files = []
        for patch_file in patch_names_list:
            file_name_split = patch_file.split('.')
            if file_name_split[1] == ('json'):
                print(patch_file)
                patch_path = os.path.join(root_folder, patch_file)
                print("loading file ",os.path.abspath(patch_path))
                with open(os.path.abspath(patch_path)) as f:
                    patch = json.load(f)
                    raw_location = patch["bounding_boxes"][0]["raw_location"]
                    coordinates_string = raw_location[10:].split(',')
                    coordinates = coordinates_string[0].split(' ')
                    lt_ln_data[file_name_split[0]] = {"latitude":coordinates[1], "longitude":coordinates[0],
                                                      "timestamp":patch["timestamp"], "country_code":patch["country_code"]}
            else:
                image_files.append({file_name_split[0]:patch_file})
        for image_file in image_files:
            ltln_data = lt_ln_data[list(image_file.keys())[0]]
            csvwriter.writerow(
                [list(image_file.values())[0], ltln_data["latitude"], ltln_data["longitude"], ltln_data["timestamp"],
                 ltln_data["country_code"]])

# Parse the BigEarthNet dataset metadata files and create a csv file indicating image name, converted location, and timestamp
def parse_bigenet_dataset(root_folder = "/workspace/app/data/raw/BigEarthNet-v1.0", csv_file="train-testing.csv"):
    folder_path_list = []
    csv_file_to_parse = os.path.join("/workspace/app/data/raw/bigearthnet-models/splits", csv_file)
    if not os.path.exists(csv_file_to_parse):
        print('ERROR: file', csv_file_to_parse, 'does not exist')

    splits = glob(f"{csv_file_to_parse}")
    patch_names_list = []
    for file in splits:
        print(file)
        with open(file, 'r') as fp:
            csv_reader = csv.reader(fp, delimiter=',')
            for row in csv_reader:
                patch_names_list.append(row[0].strip())
    print(patch_names_list)

    # writing to csv file
    output_file = csv_file.split('.')[0]+"_lat_lon_time.csv"
    with open(output_file, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # Parse each json file
        lt_ln_data = {}
        for patch_file in patch_names_list:
            print(patch_file)
            patch_path = os.path.join(root_folder, patch_file,patch_file+"_labels_metadata.json")
            print("loading file ",patch_path)
            with open(patch_path) as f:
                patch = json.load(f)
                utm_coordinates = patch["coordinates"]
                ulx = utm_coordinates["ulx"]
                uly = utm_coordinates["uly"]
                projection = patch["projection"].split(",")[0]
                zone = projection.split(" ")[-1].strip()[:-1]
                # The value will be 29N. So, remove the last part
                zone = zone[:-2]
                #print("Zone: ",zone)
                latitude, longitude = convert_utm_to_latlng(int(zone), ulx, uly)
                #print("coordinates",latitude, longitude)
                #print("acquisition_date", patch["acquisition_date"])
                lt_ln_data[patch_file] = {"latitude":latitude, "longitude":longitude,
                                                  "timestamp":patch["acquisition_date"]}

        for patch_file in patch_names_list:
            ltln_data = lt_ln_data[patch_file]
            csvwriter.writerow(
                [patch_file, ltln_data["latitude"], ltln_data["longitude"], ltln_data["timestamp"]])


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(
        description='This script provides multiple util functions which are used for data creation/processing.')
    parser.add_argument('-cull', '--convert_utm_to_latlng', default=False, type=str2bool,
                        help="whether to convert UTM to latitute and longitude")
    parser.add_argument('-z', '--zone', type=int,
                        help="what's the zone? Ex:29")
    parser.add_argument('-e', '--easting', type=int,
                        help="what's the easting? Ex: 540780")
    parser.add_argument('-n', '--northing', type=int,
                        help="what's the easting? Ex: 4312440")

    # parse fmow dataset
    parser.add_argument('-pfm', '--parsefmow', default=False, type=str2bool,
                        help="whether parse fmow file")
    parser.add_argument('-pfmfd', '--parsefmowfolder', default="/workspace/app/data/raw/fMoW/fmow-rgb", type=str,
                        help="fmow file")
    parser.add_argument('-pfmfl', '--parsefmowfile', type=str,
                        help="fmow file")

    # parse_bigenet_dataset
    parser.add_argument('-pben', '--parseben', default=False, type=str2bool,
                        help="whether parse fmow file")
    parser.add_argument('-pbenfd', '--parsebenfolder', default="/workspace/app/data/raw/BigEarthNet-v1.0", type=str,
                        help="fmow file")
    parser.add_argument('-pbenfl', '--parsebenfile', type=str,
                        help="bigearthnet file (train.csv/val.csv/test.csv)")

    args = parser.parse_args()

    if args.convert_utm_to_latlng:
        print('convert_utm_to_latlng---START')
        print(convert_utm_to_latlng(args.zone, args.easting, args.northing))
        print('convert_utm_to_latlng---END')

    if args.parsefmow:
        print('parse_fmow_dataset---START')
        print(parse_fmow_dataset(root_folder=args.parsefmowfolder, csv_file=args.parsefmowfile))
        print('parse_fmow_dataset---END')

    if args.parseben:
        print('parse_bigenet_dataset---START')
        print(parse_bigenet_dataset(root_folder=args.parsebenfolder, csv_file=args.parsebenfile))
        print('parse_bigenet_dataset---END')
