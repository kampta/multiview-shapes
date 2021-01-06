import os
import shutil
from multiprocessing import Pool
from functools import partial
import argparse

import cv2
import torch
import numpy as np
import json

category_mapping = {'02691156': 'airplane', '03001627': 'chair', '04256520': 'sofa', '04379243': 'table', '02958343': 'car'}


def run_process(args, cat_id, model_id):
	model_path = os.path.join(args.shapenet_path, cat_id, model_id, "models", "model_normalized.obj")
	output_dir = os.path.join(args.output_path, cat_id)
	try:
		command = "blender --background --python " + args.render_script_path + " --  --obj " + model_path + " --output_folder " + output_dir + " --res "+str(args.resolution) + " --camera_path "+str(args.camera_path) +" --cam_type " + args.cam_type
		os.system(command)
		load_path = os.path.join(output_dir, model_id)
		depthmaps = [cv2.imread(os.path.join(load_path, 'depth_' + str(i).zfill(2) + '0001.png')) for i in range(20)]
		depthmaps = [-torch.from_numpy(np.float32(cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY) / 255.0) * 2 - 1).float() for depthmap in depthmaps]
		torch.save(torch.stack(depthmaps), os.path.join(args.output_path, category_mapping[cat_id], model_id + '.pt'))
		shutil.rmtree(load_path)
	except Exception as e:
		print("Failed for the model: {} with error {} ", model_id, e)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Render depthmaps for shapenet models')
	parser.add_argument('--shapenet_path', type=str, default="../data/ShapeNetCore.v2/", help='path to the shapenet models')
	parser.add_argument('--output_path', type=str, default="../data/blender_data")
	parser.add_argument('--resolution', type=str, default=64, help='resolution of the resultant images')
	parser.add_argument('--categories', type = str, nargs="*", default = ['02691156', '03001627', '04256520', '04379243'], help='category model ids to render the depth maps for. Make sure you specify the ids within single quotes for windows')
	parser.add_argument('--all_categories', action='store_true', help='Run for all categories')
	parser.add_argument('--render_script_path', type=str, default="./blender_script.py", help='path to the pyblender script to generate depth maps.')
	parser.add_argument('--nproc', type=int, default=16, help='Number of threads for multithreading')
	parser.add_argument('--taxonomy_path', type=str, default='./category_mapping.json', help="shapenet taxonomy file path - required to render all categories")
	parser.add_argument('--cam_type', type=str, default="PERSP", help = "PERSP or ORTHO")
	parser.add_argument('--camera_path', type=str, default='./camPosListDodecAzEl.txt', help='path to the file with viewpoints')
	args = parser.parse_args()

	resolution = args.resolution

	if args.all_categories:
		category_mapping = json.load(open(args.taxonomy_path))	
		args.categories = os.listdir(args.shapenet_path)		

	if not os.path.exists(args.output_path):
		print("The directory " + args.output_path + " does not exist. Creating new directory...")
		os.mkdir(args.output_path)

	args.output_path = os.path.join(args.output_path, "blender_depthmaps_"+str(resolution)+"x"+str(resolution))

	tensor_dir = args.output_path
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)

	for cat in args.categories:
		if not os.path.isdir(os.path.join(args.shapenet_path, cat)):
			continue
		if not os.path.exists(os.path.join(args.output_path, cat)):
			os.mkdir(os.path.join(args.output_path, cat))
		if os.path.exists(os.path.join(args.output_path, category_mapping[cat])):
			continue
		else:
			os.mkdir(os.path.join(args.output_path, category_mapping[cat]))
		models = os.listdir(os.path.join(args.shapenet_path, cat))
		func = partial(run_process, args, cat)
		with Pool(processes=args.nproc) as pool:
			pool.map(func, models)
		shutil.rmtree(os.path.join(args.output_path, cat))
