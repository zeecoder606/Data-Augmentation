#!/bin/bash

python compute_coordinates_train.py
python create_pairs_dataset_train.py
python generate_pose_map_train.py