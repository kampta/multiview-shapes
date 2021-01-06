import numpy as np
from scipy.spatial import cKDTree as KDTree
import math
import argparse 


# ref: https://github.com/facebookresearch/DeepSDF/blob/master/deep_sdf/metrics/chamfer.py
# takes one pair of reconstructed and gt point cloud and return the cd
def compute_cd(gt_points, gen_points):

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


if __name__ == '__main__':

    num_gen_pts_sample = 30000

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_pts_path', type=str, help='Path to ground truth point clouds (numpy array)')
    parser.add_argument('--gen_pts_path', type=str, help='Path to corresponsing reconstructed point clouds (numpy array)')

    args = parser.parse_args()

    test_gt_pts = np.load(args.gt_pts_path, allow_pickle=True)
    test_gen_pts = np.load(args.gen_pts_path, allow_pickle=True)

    assert test_gen_pts.shape[0] == test_gt_pts.shape[0]

    num_instances = test_gen_pts.shape[0]
    chamfer_results = []
    
    print('Might take a few minutes ...')
    
    for instance_idx in range(num_instances):

        gt_pts_instance = test_gt_pts[instance_idx]
        gen_pts_instance = test_gen_pts[instance_idx]
        
        if gen_pts_instance.shape[0] < 2000:
            continue
        
        # if the number of points in reconstructed point cloud is < num_gen_pts_sample, 
        # repeat the points randomly to make number of points = num_gen_pts_sample
        if gen_pts_instance.shape[0]<num_gen_pts_sample:
            pt_indices = np.concatenate([
            np.arange(len(gen_pts_instance)),
            np.random.choice(len(gen_pts_instance), num_gen_pts_sample-len(gen_pts_instance))
            ])
            gen_pts_instance = gen_pts_instance[pt_indices]

        np.random.shuffle(gt_pts_instance)
        np.random.shuffle(gen_pts_instance)

        cd = compute_cd(gt_pts_instance, gen_pts_instance)

        if math.isnan(cd):
            continue
        chamfer_results.append(cd)

    chamfer_results.sort()
    print('Ground truth point cloud:  {}'.format(args.gt_pts_path))
    print('Reconstructed point cloud: {}'.format(args.gen_pts_path))
    cd_avg = sum(chamfer_results) / float(len(chamfer_results))
    print('Average Chamfer Distance: {}'.format(cd_avg))
    print('Median Chamfer Distance: {}'.format(chamfer_results[len(chamfer_results)//2]))
    print('-'*80)
