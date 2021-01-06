import sys
import os
from io import BytesIO
import csv
from pathlib import Path
import lmdb
from PIL import Image
# import cv2
import numpy as np

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets

EPS = 1e-4


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_tensor_loader(loader_type, view_id=0, load_sil=True, input_type='silhouette'):
    if loader_type == '1':
        def tensor_loader(path):
            img = torch.load(path)
            if len(img) > 20:
                depth_map = img[:20]
                sil_map = img[20:, :, :]

                # Scale depth maps to values in range [0,1]
                depth_map = (depth_map - 1.5) / 2.0

                sil_map = (sil_map > 0.5).type(torch.FloatTensor)
                img = depth_map * sil_map + (-1.0 * (1 - sil_map))
            return img

    elif loader_type == '2':
        def tensor_loader(path):
            img = torch.load(path)
            if len(img) > 20:
                depth_map = img[:20]
                sil_map = img[20:, :, :]

                # Scale depth maps to values in range [0,1]
                depth_map = (depth_map - 1.5) / 2.0

                sil_map = (sil_map > 0.5).type(torch.FloatTensor)
                img = sil_map - depth_map * sil_map
            return img

    elif loader_type == '3':
        def tensor_loader(path):
            img = torch.load(path)
            if len(img) > 20:
                depth_map = img[view_id]
                sil_map = img[20 + view_id]
                depth_map = (depth_map - 1.5) / 2.0
                sil_map = (sil_map > 0.5).type(torch.FloatTensor)
                img = depth_map * sil_map + (-1.0 * (1 - sil_map))
                return torch.stack((img, sil_map), dim=0)
            return None

    elif loader_type == 'vp':
        def tensor_loader(path):
            img = torch.load(path)
            depth_map = img[0]
            sil_map = img[1]
            depth_map = (depth_map - 1.5) / 2.0
            sil_map = (sil_map > 0.5).type(torch.FloatTensor)
            img = depth_map * sil_map + (-1.0 * (1 - sil_map))
            return torch.stack((img, sil_map), dim=0)

    elif loader_type == 'blender' or loader_type == 'ortho':
        # Loader to load  blender depth maps into a tensor
        def tensor_loader(path):
            depthmaps = torch.load(path)
            sil_bin = (depthmaps > -1 + EPS).float()
            if load_sil:
                # sil = (sil_bin - 0.5) * 2.0  # Bring sil between -1 and 1
                return torch.stack((depthmaps, sil_bin), dim=0).permute(1, 0, 2, 3).contiguous()
            else:
                return depthmaps.unsqueeze(1).contiguous()

    elif loader_type == 'merged':
        def tensor_loader(path):
            depthmaps = torch.load(path)
            persp = depthmaps[:20]
            ortho = depthmaps[20:]

            # Convert perspective to only silhouette
            if input_type == 'silhouette':
                inp = (persp > -1 + EPS).float()
            elif input_type == 'depth':
                inp = persp
            elif input_type == 'ortho_silhouette':
                inp = (ortho > -1 + EPS).float()
            elif input_type == 'ortho_depth':
                inp = ortho

            # Convert ortho to depth+sil
            ortho_sil = (ortho > -1 + EPS).float()

            # Stack them all together
            return torch.stack((inp, ortho, ortho_sil), dim=0).permute(1, 0, 2, 3).contiguous()

    elif loader_type == 'gp':
        # General purpose tensor loader;
        # I don't want to keep multiple views in multiple directories.
        # In long run, I think it might be useful to keep multiple views in same directory
        # Also normalizing both depthmaps and silhouettes between -1 and 1
        def tensor_loader(path):
            img = torch.load(path)
            img = img.float()

            # Depthmaps are between 1.5 and 3.5
            depth = img[:20]
            depth = 3.5 - depth  # Bring them between 2 and 0

            # Silhouettes are between 0 and 1
            sil = img[20:]
            sil_bin = (sil > 0.5).float()
            sil = (sil_bin - 0.5) * 2.0  # Bring sil between -1 and 1

            # Bring them between 1 and -1
            # Closest point is 1 and farthest point + wall is -1
            depth = depth * sil_bin - 1

            if load_sil:
                # return 20 x 2 x size x size vector
                # All depth, sil pairs are between -1 and 1
                return torch.stack((depth, sil), dim=0).permute(1, 0, 2, 3).contiguous()
            else:
                # return 20 x 1 x size x size vector
                return depth.unsqueeze(1).contiguous()
    else:
        print("Unknown loader type %s. Exiting ..." % loader_type)
        sys.exit(0)

    return tensor_loader


class ToDiscrete(nn.Module):
    """Implement label smoothing."""

    def __init__(self, bins, smoothing=0.0):
        super(ToDiscrete, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.bins = bins

    def forward(self, x):
        """
        :param x: tensor with dimension opt(batch x _ x C x H x W (where C == 2)
        :return:
        """
        assert len(x.shape) >= 3 and x.shape[-3] == 2
        *other_dims, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)

        depth = x[:, 0]
        sil = x[:, 1] > 0
        depth = 0.5 * (depth + 1)

        # rescale all other values to 1 and bins-1 (leave 0 for bg)
        depth = (1 + torch.round(depth * (self.bins - 2))).long()
        depth[~sil] = 0  # bg is at 0 index

        disc_x = torch.zeros(x.shape[0], self.bins, H, W, dtype=torch.float,
                             device=x.device).fill_(self.smoothing / self.bins)
        disc_x.scatter_(1, depth.data.unsqueeze(1), self.confidence)

        disc_x = disc_x.reshape(other_dims + [self.bins, H, W])
        return disc_x


class ToContinuous(nn.Module):

    def __init__(self):
        super(ToContinuous, self).__init__()

    def forward(self, x):
        """
        :param x: tensor with dimension opt(batch x _ x bins x H x W
        :return:
        """
        assert len(x.shape) >= 3 and x.shape[-3] >= 2
        *other_dims, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = torch.max(x, dim=1).indices

        sil = x > 0
        sil_float = sil.float()

        x = (x.float() - 1) / (C - 2)  # between 0 and 1

        x = 2 * x - 1
        x[~sil] = -1
        # sil_float[~sil] = -1.

        x = torch.stack((x, sil_float), dim=0).permute(1, 0, 2, 3).contiguous()
        x = x.reshape(other_dims + [2, H, W])
        return x


class DepthMapDataset(datasets.DatasetFolder):

    def __init__(self, root, loader_type='gp', categories=None, load_sil=True, view_id=0, input_type='silhouette'):
        loader = get_tensor_loader(loader_type, view_id=view_id, load_sil=load_sil, input_type=input_type)
        super(DepthMapDataset, self).__init__(
            root, loader, extensions=('.pt',), transform=None, target_transform=None)

        if categories is not None and len(categories) > 0:
            cat_to_idx = {categories[i]: i for i in range(len(categories))}
            self.samples = [
                (s[0], cat_to_idx[self.classes[s[1]]])
                for s in self.samples if self.classes[s[1]] in categories
            ]
            self.targets = [s[1] for s in self.samples]
            self.classes = categories
            self.class_to_idx = cat_to_idx

        print("%d examples across %d categories" % (len(self.samples), len(self.classes)))


class LatentCodeDataset(Dataset):
    def __init__(self, root, min_max=None, device='cuda'):
        self.data = torch.from_numpy(np.load(root)).type(torch.FloatTensor)
        self.min_max = min_max
        if min_max is None:
            data_max, _ = torch.max(self.data, dim=0)
            data_min, _ = torch.min(self.data, dim=0)
            self.min_max = [data_min, data_max]
            # print(torch.min(min_max[0]), torch.max(min_max[0]))
        self.data = (self.data - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
        # self.data = torch.clamp(self.data, 0, 1)

    def get_min_max(self):
        return self.min_max

    def __len__(self):
        return self.data.shape[0]

    def get_dim(self):
        return self.data.shape[-1]

    def __getitem__(self, item):
        return self.data[item]


def get_model_id(path):
    return Path(path).stem


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, model_ids):
        self.dataset = dataset
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.indices = [
            idx for idx, sample in enumerate(dataset.samples)
            if get_model_id(sample[0]) in model_ids
        ]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def shapenet_splits(dataset):
    r"""
    Split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
         dataset (Dataset): Dataset to be split
    """
    train = set()
    valid = set()
    test = set()

    with open('train_val_split.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == 'train':
                train.add(row['modelId'])
            elif row['split'] == 'val':
                valid.add(row['modelId'])
            elif row['split'] == 'test':
                test.add(row['modelId'])

    return [Subset(dataset, train), Subset(dataset, valid), Subset(dataset, test)]


def load_merge_and_write(persp_file, ortho_file, dest_file):
    persp = torch.load(persp_file)
    ortho = torch.load(ortho_file)
    torch.save(torch.cat([persp, ortho]), dest_file)


def merge_perspective_ortho(persp_path, ortho_path, dest_path, num_threads=1):
    from glob import glob
    from tqdm import tqdm

    persp_files = glob(persp_path + '/*/*.pt')
    persp_file_names = [f.split(persp_path)[-1] for f in persp_files]
    print(f'{len(persp_file_names)} perspective files')

    ortho_files = glob(ortho_path + '/*/*.pt')
    ortho_file_names = [f.split(ortho_path)[-1] for f in ortho_files]
    print(f'{len(ortho_file_names)} orthographic files')

    common_files = set(persp_file_names) & set(ortho_file_names)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if num_threads > 1:
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(num_threads)
        results = []

    for common_file in tqdm(common_files):
        # common file will be something like '/bench/fa1ab735efa7255c81553c4a57179bef.pt'
        _, category, fname = common_file.split('/')

        out_dir = dest_path + '/' + category
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        persp_file = persp_path + '/' + common_file
        ortho_file = ortho_path + '/' + common_file
        dest_file = dest_path + '/' + common_file
        if num_threads > 1:
            results.append(pool.apply_async(load_merge_and_write, args=(persp_file, ortho_file, dest_file)))
        else:
            load_merge_and_write(persp_file, ortho_file, dest_file)

    if num_threads > 1:
        for r in tqdm(results):
            _ = r.get()

        pool.close()  # call when you're never going to submit more work to the Pool instance
        pool.join()  # wait for the worker processes to terminate


def main():
    import argparse
    from torch.utils import data
    from torchvision.utils import save_image

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Data loader check')
    parser.add_argument('path', type=str)
    parser.add_argument('--categories', default=[], nargs='+')
    parser.add_argument('--loader_type', default='VON', help='VON or blender')
    parser.add_argument('--input_type', default='silhouette', help='silhouette or depth')
    parser.add_argument('--load_sil', action='store_true')  # default value is false
    parser.add_argument('--bins', type=int, default=0)  # default value is 0
    parser.add_argument('--smoothing', type=float, default=0.2)  # default value is 0

    # For data merging
    parser.add_argument('--ortho', type=str, default=None)
    parser.add_argument('--persp', type=str, default=None)
    parser.add_argument('--threads', type=int, default=1)  # default value is 0

    args = parser.parse_args()

    if (args.ortho is not None) or (args.persp is not None):
        merge_perspective_ortho(args.persp, args.ortho, args.path, num_threads=args.threads)
        return

    dataset = DepthMapDataset(
        args.path, categories=args.categories, loader_type=args.loader_type,
        load_sil=args.load_sil, view_id=0
    )
    train, valid, test = shapenet_splits(dataset)
    print(f"length of train/valid/test: {len(train)}, {len(valid)}, {len(test)}")
    loader = data.DataLoader(train, batch_size=8)

    to_discrete = ToDiscrete(args.bins, smoothing=0.2)
    to_continuous = ToContinuous()

    for X, y in loader:
        X = X.cuda()
        B, V, C, H, W = X.shape
        print(B, V, C, H, W)

        if args.bins > 0:
            disc_x = to_discrete(X)
            cont_x = to_continuous(disc_x)

        for b in range(B):
            print(b, torch.min(X[b]), torch.max(X[b]))
            save_image(X[b, :, 0].reshape(-1, 1, H, W), f'obj_{b}_dm.png', nrow=10, normalize=True)
            save_image(X[b, :, 1].reshape(-1, 1, H, W), f'obj_{b}_sil.png', nrow=10, normalize=True)

            if args.bins > 0:
                save_image(cont_x[b, :, 1].reshape(-1, 1, H, W), f'obj_{b}_disc_sil.png', nrow=10, normalize=True)
                save_image(cont_x[b, :, 0].reshape(-1, 1, H, W), f'obj_{b}_disc.png', nrow=10, normalize=True)
        break
