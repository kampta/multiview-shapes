import argparse
import math
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from tqdm import tqdm
from utils import gen_colors, count_parameters, depth2cloud, read_camera_positions

try:
    import wandb
except ImportError:
    wandb = None

from model import Generator, Encoder
from dataset import DepthMapDataset, shapenet_splits, seed_torch, ToDiscrete, ToContinuous
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def kldiv_loss(pred, real):
    # shape of both pred and real is B x bins x H x W
    C = pred.shape[1]
    pred = F.log_softmax(pred.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)
    real = real.permute(0, 2, 3, 1).reshape(-1, C)
    loss = F.kl_div(pred, real, reduction='batchmean')

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[-1] * fake_img.shape[-2]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, train_loader, val_loader, generator, encoder, g_ema, g_optim, d_optim, device):
    max_cat = len(train_loader.dataset.classes)
    train_loader = sample_data(train_loader)
    max_vps = args.max_vps
    # num_nbrs = reproj_consist.num_nbrs
    if args.bins > 1:
        criterion = kldiv_loss
    elif args.soft_l1:
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.L1Loss()
    colors = [torch.tensor(c, device=device) for c in gen_colors(max_vps)]
    to_discrete = ToDiscrete(args.bins, smoothing=args.smoothing)
    to_continuous = ToContinuous()
    wc2cc, cc2wc = read_camera_positions('camPosListDodecAzEl.txt', device, args.loader_type)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0

    if args.distributed:
        g_module = generator.module
        e_module = encoder.module

    else:
        g_module = generator
        e_module = encoder

    accum = args.decay  # 0.5 ** (32 / (10 * 1000))

    n_fixed_samples = min(8 * max_cat, args.batch if args.val_batch is None else args.val_batch)
    fixed_object, fixed_cat = next(iter(val_loader))
    fixed_object, fixed_cat = fixed_object[:n_fixed_samples], fixed_cat[:n_fixed_samples]

    fixed_vp_in = torch.tensor(np.random.choice(max_vps, n_fixed_samples))
    fixed_cat_in = fixed_cat
    fixed_cat_out = torch.repeat_interleave(fixed_cat, max_vps)
    fixed_input = fixed_object[np.arange(n_fixed_samples), fixed_vp_in]
    if args.bins > 1 and args.loader_type != 'merged':
        fixed_dm_in = to_discrete(fixed_input)
    elif args.bins > 1 and args.loader_type == 'merged' and args.input_quant:
        fixed_dm_in = fixed_input[:, [0]]
        fixed_sil = (fixed_dm_in > -1).float()
        fixed_dm_in = to_discrete(torch.cat((fixed_dm_in, fixed_sil), dim=1))
    elif args.loader_type == 'merged':
        fixed_dm_in = fixed_input[:, [0]]
    else:
        fixed_dm_in = fixed_input
    fixed_dm_in, fixed_vp_in, fixed_cat = fixed_dm_in.to(device), fixed_vp_in.to(device), fixed_cat.to(device)

    if args.loader_type == 'merged':
        # Since first 20 are the inputs, we can directly index them
        fixed_object = fixed_object[:, :, [1, 2]]

    if args.wandb:
        # Render fixed output point clouds
        fixed_pc = [
            depth2cloud(fixed_object[b, v, 0, :, :], cc2wc[v].cpu(),
                        args.size, fixed_object[b, v, 1, :, :], data_type='ortho')
            for b in range(n_fixed_samples) for v in range(max_vps)
        ] if args.load_sil else [
            depth2cloud(fixed_object[b, v, 0, :, :], cc2wc[v].cpu(),
                        args.size, data_type='ortho')
            for b in range(n_fixed_samples) for v in range(max_vps)
        ]
        # Add colors
        fixed_pc = [
            torch.cat([fixed_pc[ii], torch.zeros_like(fixed_pc[ii]) + colors[ii % max_vps].cpu()], dim=1)
            for ii in range(len(fixed_pc))
        ]
        fixed_pc = [torch.cat(fixed_pc[b * max_vps:(b + 1) * max_vps]) for b in range(n_fixed_samples)]
        fixed_pc = [wandb.Object3D(
            fixed_pc[b][:, [2, 0, 1, 3, 4, 5]].data.numpy(),
            caption="Object_%02d" % b
        ) for b in range(n_fixed_samples)]

        # Render inputs
        fixed_im = [wandb.Image(
            fixed_input[ii, 0].data.numpy(),
            caption="Object_%02d_%02d" % (ii, vp.item()))
            for ii, vp in enumerate(fixed_vp_in)
        ]

        wandb.log({
            "Fixed Input": fixed_im,
            "Fixed PC": fixed_pc,
        }, step=0)

    requires_grad(generator, True)
    requires_grad(encoder, True)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        ## Training
        objects, cats = next(train_loader)
        # objects, cats = objects.to(device), cats.to(device)

        B = objects.shape[0]
        vp_in_out = torch.tensor(
            np.random.choice(max_vps, B * args.num_vps * 2).reshape(-1, 2)
        )
        vp_in = vp_in_out[:, 0]
        vp_out = vp_in_out[:, 1]
        batch_ids = np.repeat(np.arange(B), args.num_vps)

        if args.loader_type == 'merged':
            objects_in = objects[:, :, [0]]
            objects_out = objects[:, :, [1, 2]]
        else:
            objects_in = objects_out = objects

        dm_in = objects_in[batch_ids, vp_in]
        dm_out = objects_out[batch_ids, vp_out]
        cats = torch.repeat_interleave(cats, repeats=args.num_vps)
        dm_in, dm_out, vp_in, vp_out, cats = \
            dm_in.to(device), dm_out.to(device), vp_in.to(device), vp_out.to(device), cats.to(device)

        if args.bins > 1 and args.loader_type != 'merged':
            dm_in = to_discrete(dm_in)
            dm_out = to_discrete(dm_out)
        elif args.bins > 1 and args.input_quant:
            # dm_in = to_discrete(dm_in)
            dm_sil = (dm_in > -1).float()
            dm_in = to_discrete(torch.cat((dm_in, dm_sil), dim=1))
            dm_out = to_discrete(dm_out)
        elif args.bins > 1:
            dm_out = to_discrete(dm_out)

        styles = encoder(dm_in, viewpoints=vp_in, categories=cats)

        # Take average of latents of viewpoints to retain only style info
        styles = styles.reshape(B, args.num_vps, -1)
        styles = torch.mean(styles, dim=1).unsqueeze(1)
        styles = styles.expand(-1, args.num_vps, -1).reshape(B * args.num_vps, -1)

        # Reproject this style in new viewpoints
        reprojections, _ = generator([styles], viewpoints=vp_out, categories=cats)

        loss = criterion(reprojections, dm_out)
        loss_dict['reproj_loss_train'] = loss

        d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
        if d_regularize:
            dm_in.requires_grad = True
            r1_loss = d_r1_loss(styles, dm_in)

            loss += args.r1 / 2 * r1_loss * args.d_reg_every + 0 * styles
        loss_dict['r1'] = r1_loss

        g_regularize = args.g_reg_every > 0 and i % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, B // args.path_batch_shrink)
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                reprojections[:path_batch_size], styles[:path_batch_size], mean_path_length
            )
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * reprojections[0, 0, 0, 0]

            loss += weighted_path_loss
            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )
        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)

        reproj_loss_train = loss_reduced['reproj_loss_train'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        encoder.zero_grad()
        generator.zero_grad()
        loss.backward()
        d_optim.step()
        g_optim.step()

        if get_rank() == 0:
            pbar.set_description((
                f'reproj_loss_train: {reproj_loss_train:.4f}; '
                # f'r1: {r1_val:.4f}; '
                # f'path: {path_loss_val:.4f}; '
                # f'mean path: {mean_path_length_avg:.4f}, '
            ))

            if wandb and args.wandb:
                wandb.log(
                    {
                        'Train reprojection': reproj_loss_train,
                    }, step=i
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    generator.eval()
                    encoder.eval()

                    reproj_loss_val = torch.tensor(0.0)
                    val_count = 0
                    for objects, cats in val_loader:
                        # Reshape appropriately
                        # B, V, C, H, W = objects.shape
                        B = objects.shape[0]
                        objects, cats = objects.to(device), cats.to(device)

                        if args.loader_type == 'merged':
                            objects_in = objects[:, :, [0]]
                            objects_out = objects[:, :, [1, 2]]
                        else:
                            objects_in = objects_out = objects

                        batch_ids = np.repeat(np.arange(B), max_vps)
                        vp_in_out = torch.tensor(B * list(range(max_vps))).to(device)
                        dm_in = objects_in[batch_ids, vp_in_out]
                        dm_out = objects_out[batch_ids, vp_in_out]
                        dm_in, dm_out, vp_in, vp_out, cats = \
                            dm_in.to(device), dm_out.to(device), vp_in.to(device), vp_out.to(device), cats.to(device)

                        # if args.bins > 1:
                        # 	dm_out = to_discrete(dm_out)

                        if args.bins > 1 and args.loader_type != 'merged':
                            dm_in = to_discrete(dm_in)
                            dm_out = to_discrete(dm_out)
                        elif args.bins > 1 and args.input_quant:
                            dm_sil = (dm_in > -1).float()
                            dm_in = to_discrete(torch.cat((dm_in, dm_sil), dim=1))
                            # dm_in = to_discrete(dm_in)
                            dm_out = to_discrete(dm_out)
                        elif args.bins > 1:
                            dm_out = to_discrete(dm_out)

                        cats = torch.repeat_interleave(cats, repeats=max_vps)
                        styles = encoder(dm_in, viewpoints=vp_in_out, categories=cats)

                        # Take average of latents of viewpoints to retain only style info
                        styles = styles.reshape(B, max_vps, -1)
                        styles = torch.mean(styles, dim=1).unsqueeze(1)
                        styles = styles.expand(-1, max_vps, -1).reshape(B * max_vps, -1)

                        reprojections, _ = g_ema([styles], viewpoints=vp_in_out, categories=cats)

                        reproj_loss_val += criterion(reprojections, dm_out)
                        val_count += 1

                        # Do full validation every 5000 steps, else do partial
                        if i % 5000 == 0 and i > 1:
                            continue

                        if val_count > 5:
                            break

                    reproj_loss_val /= val_count
                    reproj_loss_val = reproj_loss_val.item()
                    # loss_dict['reproj_loss_val'] = reproj_loss_val
                    print(f'reproj_loss_valid: {reproj_loss_val:.4f}; ')

                    if args.wandb:
                        wandb.log({
                            "Valid reprojection": reproj_loss_val,
                        }, step=i)
                        styles = encoder(fixed_dm_in, viewpoints=fixed_vp_in, categories=fixed_cat_in)
                        styles = styles.unsqueeze(1)
                        styles = styles.expand(-1, max_vps, -1).reshape(n_fixed_samples * max_vps, -1)
                        fixed_vp_out = torch.tensor(n_fixed_samples * list(range(max_vps))).to(device)

                        reprojections, _ = g_ema([styles], viewpoints=fixed_vp_out, categories=fixed_cat_out)
                        if args.bins > 1:
                            reprojections = to_continuous(reprojections)

                        # PC reconstructions for val data
                        recon_pc = [
                            depth2cloud(reprojections[ii, 0, :, :], cc2wc[vp], args.size,
                                        reprojections[ii, 1, :, :], data_type=args.loader_type)
                            for ii, vp in enumerate(fixed_vp_out)
                        ] if args.load_sil else [
                            depth2cloud(reprojections[ii, 0, :, :], cc2wc[vp], args.size,
                                        data_type=args.loader_type)
                            for ii, vp in enumerate(fixed_vp_out)
                        ]
                        # Add colors
                        recon_pc = [
                            torch.cat([recon_pc[ii], torch.zeros_like(recon_pc[ii]) + colors[vp]], dim=1)
                            for ii, vp in enumerate(fixed_vp_out)
                        ]
                        # print(len(recon_pc), B, max_vps, num_nbrs)
                        recon_pc = [torch.cat(recon_pc[b * max_vps:(b + 1) * max_vps]) for b in range(n_fixed_samples)]
                        recon_pc = [wandb.Object3D(
                            recon_pc[b][:, [2, 0, 1, 3, 4, 5]].data.cpu().numpy(),
                            caption="Object_%02d" % b
                        ) for b in range(n_fixed_samples)]

                        wandb.log({
                            "Recon PC": recon_pc,
                        }, step=i)

                torch.save({
                    'g': g_module.state_dict(),
                    'd': e_module.state_dict(),
                    'g_ema': g_ema.state_dict(),
                    'g_optim': g_optim.state_dict(),
                    'd_optim': d_optim.state_dict(),
                },
                    args.ckpt_save_directory + f'/{str(i).zfill(6)}.pt',
                )
                requires_grad(generator, True)
                requires_grad(encoder, True)
                encoder.train()
                generator.train()


def main():
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('--decay', type=float, default=0.5 ** (32 / (10 * 1000)))
    parser.add_argument('--view_id', type=int, default=0)
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=None)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)
    # parser.add_argument('--n_sample', type=int, default=16)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--initial_size', type=int, default=4)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    # parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=float, default=2)
    parser.add_argument('--wandb', action='store_true')  # default value is true
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt_save_directory', type=str, default='checkpoint')
    parser.add_argument('--sample_save_directory', type=str, default='sample')
    parser.add_argument('--load_sil', action='store_true')  # default value is false
    parser.add_argument('--input_type', type=str, default='silhouette')  # default value is false
    parser.add_argument('--input_quant', action='store_true', help='make input quantized')  # default value is false
    parser.add_argument('--exp_name', type=str, default='0')
    parser.add_argument('--loader_type', type=str, default='gp')
    parser.add_argument('--categories', default=[], nargs='+',
                        help="what all categories from shapenet to use")
    parser.add_argument('--no_noise', action='store_true',
                        help="remove noise in StyledConv layer")
    parser.add_argument('--max_vps', type=int, default=20,
                        help="maximum viewpoints")
    parser.add_argument('--num_vps', type=int, default=2,
                        help="sample num_vps at a time")
    parser.add_argument('--soft_l1', action='store_true',
                        help="use soft l1 loss")
    parser.add_argument('--random_avg', action='store_true',
                        help="use uniformly sampled noise for averaging latent")
    parser.add_argument('--use_pretrained_if_available', action='store_true',
                        help="Use pretrained model with largest iter if available")
    parser.add_argument('--uncond_cat_enc', action='store_true', help="make encoder unconditional on category")
    parser.add_argument('--uncond_vp_enc', action='store_true', help="make encoder unconditional on viewpoint")
    parser.add_argument('--merge_conditions', action='store_true', help="merge viewpoint and category embedding")
    parser.add_argument('--bins', type=int, default=0)  # default value is 0
    parser.add_argument('--smoothing', type=float, default=0.2)  # default value is 0.2
    parser.add_argument('--seed', type=int, default=0)  # Seed for dataloader

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # args.ckpt_save_direcory = '/scratch1/jsreddy/stylegan_depthmaps_data/'+args.ckpt_save_directory + "_" + str(args.exp_num)
    args.ckpt_save_directory = os.path.join(args.ckpt_save_directory, args.exp_name)
    if not os.path.exists(args.ckpt_save_directory):
        os.makedirs(args.ckpt_save_directory)
        print("Created Ckpt Directory: ", args.ckpt_save_directory)

    # args.sample_save_directory = '/scratch1/jsreddy/stylegan_depthmaps_data/'+args.sample_save_directory + "_" + str(args.exp_num)
    args.sample_save_directory = os.path.join(args.sample_save_directory, args.exp_name)
    if not os.path.exists(args.sample_save_directory):
        os.makedirs(args.sample_save_directory)
        print("Created Sample Directory: ", args.sample_save_directory)

    # args.latent = 512
    # args.n_mlp = 8
    args.start_iter = 0
    if args.bins > 1:
        args.output_channels = args.bins
        args.load_sil = True
    elif args.load_sil:
        args.output_channels = 2
    else:
        args.output_channels = 1

    args.input_channels = args.output_channels
    if args.loader_type == 'merged':
        args.input_channels = args.bins if args.input_quant else 1

    # 3D stuff
    # reproj_consist = ReprojectionConsistency(
    # 	include_self=args.include_self, use_sil=args.load_sil, device=device, data_type=args.loader_type)

    # Dataset
    # dataset = MultiResolutionDataset(args.path, transform, args.size)
    dataset = DepthMapDataset(
        args.path, categories=args.categories, loader_type=args.loader_type,
        load_sil=args.load_sil, view_id=args.view_id, input_type=args.input_type
    )

    # Generator call
    generator = Generator(
        args.size, args.latent, args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        output_channels=args.output_channels,
        initial_size=args.initial_size,
        num_viewpoints=args.max_vps,
        no_noise=args.no_noise,
        num_categories=len(dataset.classes),
        merge_conditions=args.merge_conditions
    ).to(device)

    print("Generator:", generator)

    # Encoder call
    encoder = Encoder(
        args.size, args.latent,
        channel_multiplier=args.channel_multiplier,
        input_channels=args.input_channels,
        num_viewpoints=1 if args.uncond_vp_enc else args.max_vps,
        initial_size=args.initial_size,
        num_categories=1 if args.uncond_cat_enc else len(dataset.classes),
    ).to(device)

    print("Encoder: ", encoder)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        output_channels=args.output_channels,
        initial_size=args.initial_size,
        num_viewpoints=args.max_vps,
        no_noise=args.no_noise,
        num_categories=len(dataset.classes),
        merge_conditions=args.merge_conditions
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.g_reg_every > 0 else 1
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.d_reg_every > 0 else 1

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'])
        encoder.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

    elif args.use_pretrained_if_available:
        from glob import glob

        files = sorted(glob(os.path.join(args.ckpt_save_directory, '*.pt')))
        if len(files) > 0:
            print('found model:', files[-1])
            ckpt = torch.load(files[-1])
            ckpt_name = os.path.basename(files[-1])
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

            generator.load_state_dict(ckpt['g'])
            encoder.load_state_dict(ckpt['d'])
            g_ema.load_state_dict(ckpt['g_ema'])

            g_optim.load_state_dict(ckpt['g_optim'])
            d_optim.load_state_dict(ckpt['d_optim'])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    if get_rank() == 0 and wandb is not None and args.wandb:
        print("Using wandb")
        wandb.init(project='vp2vp', name=str(args.exp_name))
        wandb.config.update(args)

    print("# params in encoder:", count_parameters(encoder))
    print("# params in decoder:", count_parameters(generator))

    train_dataset, val_dataset, test_dataset = shapenet_splits(dataset)
    print(f"length of train/valid/test: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")
    seed_torch(args.seed)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
        drop_last=False,
        num_workers=4,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch if args.val_batch is None else args.val_batch,
        sampler=data_sampler(val_dataset, shuffle=True, distributed=args.distributed),
        drop_last=False,
        num_workers=4,
    )

    print("Training.....")
    train(args, train_loader, val_loader, generator, encoder, g_ema, g_optim, d_optim, device)


if __name__ == '__main__':
    main()
