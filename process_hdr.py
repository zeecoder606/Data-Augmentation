import cv2
import numpy as np
import os

hdr_data = os.listdir('hdr_data')
no_of_images = 10

for hdr_file in hdr_data:
	f_name = os.getcwd() + '/hdr_data/' + hdr_file
	save_name = os.getcwd() + '/ldr_images/' + hdr_file.split(".")[0]
	if not os.path.isdir(save_name):
		os.mkdir(save_name)

	HDR = cv2.imread(f_name,cv2.IMREAD_ANYDEPTH)
	for i in range(0,10):
		gamma = 0.2*(i+1)

		# gamma, intensity, light_adapt, colour_adapt
		tonemapReinhard = cv2.createTonemapReinhard(gamma,0.5,1.0,1.0)
		ldrReinhard = tonemapReinhard.process(HDR)
		
		save_img = cv2.imwrite(save_name + "/output_%d.jpg"%i, ldrReinhard * 255)
