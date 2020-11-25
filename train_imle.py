import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
import faiss
import argparse
import collections

# Code from https://github.com/yedidh/glann.git
OptParams = collections.namedtuple('OptParams', 'lr batch_size epochs ' +
                                                'decay_epochs decay_rate ')
OptParams.__new__.__defaults__ = (None, None, None, None, None)


class _netT(nn.Module):
    def __init__(self, xn, yn):
        super(_netT, self).__init__()
        self.xn = xn
        self.yn = yn
        self.lin1 = nn.Linear(xn, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.lin_out = nn.Linear(128, yn, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        z = self.lin1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.lin_out(z)
        return z


class _ICP():
    def __init__(self, e_dim, z_dim):
        self.e_dim = e_dim
        self.z_dim = z_dim
        self.netT = _netT(e_dim, z_dim).cuda()

    def train(self, z_np, opt_params):
        self.opt_params = opt_params
        for epoch in range(opt_params.epochs):
            self.train_epoch(z_np, epoch)

    def train_epoch(self, z_np, epoch):
        batch_size = self.opt_params.batch_size
        n, d = z_np.shape
        batch_n = n // batch_size
        rp = np.random.permutation(n)

        decay_steps = epoch // self.opt_params.decay_epochs
        lr = self.opt_params.lr * self.opt_params.decay_rate ** decay_steps
        optimizerT = optim.Adam(self.netT.parameters(), lr=lr,
                                betas=(0.5, 0.999), weight_decay=1e-5)
        criterion = nn.MSELoss().cuda()
        self.netT.train()

        # Generate 2N random latent vectors
        M = batch_n * 2
        e_np = np.zeros((M * batch_size, self.e_dim))
        Te_np = np.zeros((M * batch_size, self.z_dim))
        for i in range(M):
            e = torch.randn(batch_size, self.e_dim).cuda()
            y_est = self.netT(e)
            e_np[i * batch_size: (i + 1) * batch_size] = e.cpu().data.numpy()
            Te_np[i * batch_size: (i + 1) * batch_size] = y_est.cpu().data.numpy()

        # Find the nearest neighbor from the 2N points for each data point
        nbrs = faiss.IndexFlatL2(self.z_dim)
        nbrs.add(Te_np.astype('float32'))
        _, indices = nbrs.search(z_np.astype('float32'), 1)
        indices = indices.squeeze(1)


        # Start optimizing
        er = 0
        for i in range(batch_n):
            self.netT.zero_grad()
            idx_np = i * batch_size + np.arange(batch_size)
            e = torch.from_numpy(e_np[indices[rp[idx_np]]]).float().cuda()
            z_act = torch.from_numpy(z_np[rp[idx_np]]).float().cuda()
            z_est = self.netT(e)
            loss = criterion(z_est, z_act)
            loss.backward()
            er += loss.item()
            optimizerT.step()

        print("Epoch: %d Error: %f" % (epoch, er / batch_n))


class ICPTrainer():
    def __init__(self, f_np, d):
        self.f_np = f_np
        self.icp = _ICP(d, f_np.shape[1])

    def train_icp(self, args):
        uncca_opt_params = OptParams(lr=args.lr, batch_size=args.batch_size, epochs=args.n_epochs,
                                     decay_epochs=args.decay_epochs, decay_rate=args.decay_rate)
        self.icp.train(self.f_np, uncca_opt_params)



def generate_styles(args, dmin, dmax):
    mappingT = _netT(args.e_dim, args.z_dim).to(args.device)
    mappingT.load_state_dict(torch.load(args.imle_ckpt))
    mappingT.eval()
    styles = mappingT(torch.randn(args.sample_batchsize, args.e_dim).to(args.device))
    styles = styles * torch.tensor(dmax - dmin).to(args.device) + torch.tensor(dmin).to(args.device)
    return styles.float()

def main(args):
    train_data = np.load(args.path)
    dmin, dmax = np.min(train_data, axis=0), np.max(train_data, axis=0)
    if args.generate_samples and os.path.exists(args.generator_ckpt):
        generated_styles = generate_styles(args, dmin, dmax)
        np.save(generated_styles.cpu().data.numpy(), 'generated_styles.npy')
        return
    normalized_train_data = (train_data-dmin)/(dmax-dmin)
    icpt = ICPTrainer(normalized_train_data, args.e_dim)
    icpt.train_icp(args)
    torch.save(icpt.icp.netT.state_dict(), args.model_save_name)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train IMLE on the shape codes')
    parser.add_argument('--path', type=str, default='../data/shape_codes.npy', help='path to the shape codes')
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--model_ckpt_save_dir', type=str, default='./models')
    parser.add_argument('--model_save_name', type=str, default='generator.pt')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--decay_epochs', type=int, default=50)
    parser.add_argument('--decay_rate', type=float, default=0.5)

    # To generate samples from the trained model
    parser.add_argument('--generate_samples', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--z_dim', type=int, default=512, help='style code dimension')
    parser.add_argument('--generator_ckpt', type=str, default='generator.pt')
    parser.add_argument('--num_samples', type=int, default=32, help='number of style samples to generate')

    args = parser.parse_args()
    if not os.path.exists(args.model_ckpt_save_dir):
        os.mkdir(args.model_ckpt_save_dir)
    args.model_save_name = os.path.join(args.model_ckpt_save_dir, args.model_save_name)

    main(args)

