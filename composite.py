import os
import cv2
import numpy as np

background_dir = "seed_data/backgrounds"
person_dir = "results/final_honeywell/test_latest/images"
composite_result_dir = "results/final_honeywell/test_latest/composite"
if not os.path.exists(composite_result_dir):
    os.mkdir(composite_result_dir)

for bg in os.listdir(background_dir):
    bg_img = cv2.imread(os.path.join(background_dir,bg))

    for folder in os.listdir(person_dir):
        fg_img = cv2.imread(os.path.join(person_dir,folder,"FG.jpg"))
        mask = np.load(os.path.join(person_dir,folder,"mask.npy"))
        composite_image = bg_img*(1-mask)+fg_img*(mask)
        
        bg_pth = bg.split(".")[0]
        img_path = bg_pth +"_"+ folder + ".jpg"
        save_path = os.path.join(composite_result_dir, img_path)

        cv2.imwrite(save_path, composite_image)
         

    