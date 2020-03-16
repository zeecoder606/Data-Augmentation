import os
import shutil
from PIL import Image

IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
	images = []
	# assert os.path.isdir(dir), '%s is not a valid directory' % dir
	# new_root = './fashion_data'
	# if not os.path.exists(new_root):
	# 	os.mkdir(new_root)

	train_root = './fashion_data/train'
	if not os.path.exists(train_root):
		os.mkdir(train_root)

	test_root = './fashion_data/test'
	if not os.path.exists(test_root):
		os.mkdir(test_root)

	train_images = []
	train_f = open('./fashion_data/train.lst', 'r')
	for lines in train_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			train_images.append(lines)

	test_images = []
	test_f = open('./fashion_data/test.lst', 'r')
	for lines in test_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			test_images.append(lines)

	print(train_images, test_images)
	

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				break_path = path.split('/')
				break_path[5] = break_path[5].replace('_', '')
				temp = break_path[6].split('_')
				break_path[6] = ''.join(temp[0]+'_'+temp[1]+temp[2])
				check_file_name = "".join(break_path[2:])
				# path_names = path.split('/') 
				# # path_names[2] = path_names[2].replace('_', '')
				# path_names[3] = path_names[3].replace('_', '')
				# path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
				# path_names = "".join(path_names)
				# new_path = os.path.join(root, path_names)
				img = Image.open(path)
				imgcrop = img.crop((40, 0, 216, 256))
				print (".....")
				if check_file_name in train_images:
					print ("yes..........................................train")
					imgcrop.save(os.path.join(train_root, check_file_name))
				elif check_file_name in test_images:
					print ("No..........................................test")
					imgcrop.save(os.path.join(test_root, check_file_name))

make_dataset('./fashion_data/fashion')
