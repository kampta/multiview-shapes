from metrics.evaluation_metrics import emd_approx
import argparse
import numpy as np
import torch


if __name__ == '__main__':

	num_rec_pts_sample = 500

	parser = argparse.ArgumentParser()
	parser.add_argument('--gt_pts_path', type=str, help='Path to ground truth point clouds (numpy array)')
	parser.add_argument('--rec_pts_path', type=str, help='Path to corresponsing reconstructed point clouds (numpy array)')

	args = parser.parse_args()

	test_gt_pts = np.load(args.gt_pts_path, allow_pickle=True)
	test_rec_pts = np.load(args.rec_pts_path, allow_pickle=True)

	assert test_rec_pts.shape[0] == test_gt_pts.shape[0]

	num_instances = test_rec_pts.shape[0]
	gt_list = []
	rec_list = []

    print('Might take a few minutes ...')
    
	for instance_idx in range(num_instances):

		gt_pts_instance = test_gt_pts[instance_idx]
		rec_pts_instance = test_rec_pts[instance_idx]

		if rec_pts_instance.shape[0] == 0:
			continue 
    
        # if the number of points in reconstructed/ground truth point cloud is < num_gen_pts_sample, 
        # repeat the points randomly to make number of points = num_gen_pts_sample
		if rec_pts_instance.shape[0]<num_rec_pts_sample:
			pt_indices = np.concatenate([
            np.arange(len(rec_pts_instance)),
            np.random.choice(len(rec_pts_instance), num_rec_pts_sample-len(rec_pts_instance))
        	])
			rec_pts_instance = rec_pts_instance[pt_indices]
		else:
			rec_pts_instance = rec_pts_instance[:num_rec_pts_sample]

		if gt_pts_instance.shape[0]<num_rec_pts_sample:
			pt_indices = np.concatenate([
            np.arange(len(gt_pts_instance)),
            np.random.choice(len(gt_pts_instance), num_rec_pts_sample-len(gt_pts_instance))
        	])
			gt_pts_instance = gt_pts_instance[pt_indices]
		else:
			gt_pts_instance = gt_pts_instance[:num_rec_pts_sample]

		np.random.shuffle(gt_pts_instance)
		np.random.shuffle(rec_pts_instance)

		gt_list.append(np.reshape(gt_pts_instance, (1, num_rec_pts_sample, 3)))
		rec_list.append(np.reshape(rec_pts_instance, (1, num_rec_pts_sample, 3)))


	gt_pts_all = np.concatenate(gt_list)
	rec_pts_all = np.concatenate(rec_list)

	emd = emd_approx(torch.from_numpy(rec_pts_all).cuda(), torch.from_numpy(gt_pts_all).cuda())
	print(emd.mean())
	print('-'*80)











		