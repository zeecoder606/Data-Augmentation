import numpy as np
from PIL import Image
import torch
import util.util as util
from util.image_pool import ImagePool
import os

path_testK = '/home/zeeshan/ashish/Pose-Transfer/seed_data/draw/'
for heat_map in os.listdir(path_testK):
    name_heat_map = path_testK + heat_map 
    input_BP1 = np.load(name_heat_map)
    input_BP1 = input_BP1.reshape(1, input_BP1.shape[0], input_BP1.shape[1], input_BP1.shape[2])
    input_BP1 = torch.from_numpy(input_BP1)
    print(heat_map)
    print (input_BP1.shape)
    image_numpy = util.draw_pose_from_map(input_BP1)[0]
    image_pil = Image.fromarray(image_numpy)
    image_path = '/home/zeeshan/ashish/Pose-Transfer/seed_data/openpose_visual/'+ heat_map.split('.')[0]+'.jpg'
    image_pil.save(image_path)
