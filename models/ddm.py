import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
import pytorch_ssim
import numpy as np
import models
# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm
# https://github.com/IGITUGraz/WeatherDiffusion

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def sobelconv(x, kernelx, kernely):
    b, c, w, h = x.shape
    batch_list = []
    for i in range(b):
        tensor_list = []
        for j in range(c):
            sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), kernelx, padding=1)
            sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), kernely, padding=1)
            add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
            tensor_list.append(add_0)

        batch_list.append(torch.stack(tensor_list, dim=1))

    return torch.cat(batch_list, dim=0)

def noise_estimation_loss_1(model, x0, t, e, b, stage):
    gt = x0[:, 3:6, :, :]
    input = x0[:, :3, :, :]
    x0ir = x0[:, 6:, :, :]
    if x0ir.size()[1]==1:
        x0ir = x0ir.repeat(1,3,1,1)
    x00 = gt
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x =  x00 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([input,x0ir,x], dim=1), t.float(), stage)
    x_out = (x - output * (1.0 - a).sqrt())/a.sqrt()
    ssim = pytorch_ssim.SSIM(window_size = 11)
    max = x00
    lmax = F.l1_loss(x_out, max)
    lssimmax = 1-ssim(max,x_out)
    return lmax, lssimmax

def noise_estimation_loss_2(model, x0, t, e, b, stage):
    input = x0[:, :3, :, :]
    x0ir = x0[:, 6:, :, :]
    if x0ir.size()[1]==1:
        x0ir = x0ir.repeat(1,3,1,1)
    x00 = x0ir
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x =  x00 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([input,x0ir,x], dim=1), t.float(), stage)
    x_out = (x - output * (1.0 - a).sqrt())/a.sqrt()
    ssim = pytorch_ssim.SSIM(window_size = 11)
    max = x00
    lmax = F.l1_loss(x_out, max)
    lssimmax = 1-ssim(max,x_out)
    return lmax, lssimmax

def noise_estimation_loss_3(model, x0, t, e, b ,kernelx, kernely, stage):
    gt = x0[:, 3:6, :, :]
    input = x0[:, :3, :, :]
    x0ir = x0[:, 6:, :, :]
    if x0ir.size()[1]==1:
        x0ir = x0ir.repeat(1,3,1,1)
    x00 = gt        
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x =  x00 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([input,x0ir, x], dim=1), t.float(), stage)
    x_out = (x - output * (1.0 - a).sqrt())/a.sqrt()
    ssim = pytorch_ssim.SSIM(window_size = 11)
    max = torch.max(x00, x0ir)
    lmax = F.l1_loss(x_out, max)
    lssimmax = 1-ssim(max,x_out)
    gradmax = torch.maximum(sobelconv(x00,kernelx, kernely), sobelconv(x0ir,kernelx, kernely))
    grad_out = sobelconv(x_out,kernelx, kernely)
    lgradmax = F.l1_loss(grad_out, gradmax)
    lssim = lssimmax
    return lmax, lssim, lgradmax

class Diffusion_Model(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        kernelx = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
        kernely = [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.kernelx = kernelx.to(self.device)
        self.kernely = kernely.to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def load_ddm_ckpt_copy(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        pretrained_weights = checkpoint['state_dict']
        scratch_dict = self.model.state_dict()
        target_dict = {}
        for k in scratch_dict.keys():
            if k.find('down3') != -1:
                copy_k = k.replace('down3','down2')
            elif k.find('conv_in3') != -1:
                copy_k = k.replace('conv_in3','conv_in2')
            elif k.find('mid3') != -1:
                copy_k = k.replace('mid3','mid2')
            else:
                copy_k = k
            target_dict[k] = pretrained_weights[copy_k].clone()
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(target_dict, strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))
         
    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        Loss_noise = []
        Loss_ssim = []
        Loss_sum = []
        # stage 1:
        for epoch in range(self.start_epoch, self.config.training.n_epochs_stage1 + 1):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1
                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:6, :, :])
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                if self.config.data.dataset=='RSFB': stage = 5
                else: stage = 1
                lnoise, lssim = noise_estimation_loss_1(self.model, x, t, e, b, stage)
                loss = 100 * (lssim * 10 + lnoise)
                if self.step % 10 == 0:
                    Loss_noise.append(lnoise.item())
                    Loss_ssim.append(lssim.item())
                    Loss_sum.append(loss.item())
                    print(f"step: {self.step}, l: {lnoise.item()}, lssim: {lssim.item()}, lsum: {loss.item()}, data time: {data_time / (i+1)}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    filename=os.path.join(self.args.model_path, 'ckpt_CDMFusion', self.config.data.dataset + '_stage1')
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename)
                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step, stage)
        self.load_ddm_ckpt_copy(filename+'.pth.tar')
        # stage 2:
        for epoch in range(self.start_epoch, self.start_epoch + self.config.training.n_epochs_stage2 + 1):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1
                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:6, :, :])
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                if self.config.data.dataset=='RSFB': stage = 6
                else: stage = 2
                lnoise, lssim = noise_estimation_loss_2(self.model, x, t, e, b, stage)
                loss = 100 * (lssim * 10 + lnoise)
                if self.step % 10 == 0:
                    Loss_noise.append(lnoise.item())
                    Loss_ssim.append(lssim.item())
                    Loss_sum.append(loss.item())
                    print(f"step: {self.step}, l: {lnoise.item()}, lssim: {lssim.item()}, lsum: {loss.item()}, data time: {data_time / (i+1)}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    filename=os.path.join(self.args.model_path, 'ckpt_CDMFusion', self.config.data.dataset + '_stage2')
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename)
                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step, stage)
        self.load_ddm_ckpt(filename +'.pth.tar')
        # stage 3:
        for epoch in range(self.start_epoch, self.start_epoch + self.config.training.n_epochs_stage3 + 1):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1
                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:6, :, :])
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                if self.config.data.dataset=='RSFB': stage = 7
                else: stage = 3
                lnoise, lssim, lgrad = noise_estimation_loss_3(self.model, x, t, e, b ,self.kernelx, self.kernely, stage)
                loss = 100 * (lssim * 10 + lnoise + 10 * lgrad)
                if self.step % 10 == 0:
                    Loss_noise.append(lnoise.item() + 10 * lgrad.item())
                    Loss_ssim.append(lssim.item())
                    Loss_sum.append(loss.item())
                    print(f"step: {self.step}, l: {lnoise.item()}, lssim: {lssim.item()}, lgrad: {lgrad.item()}, lsum: {loss.item()}, data time: {data_time / (i+1)}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    filename=os.path.join(self.args.model_path, 'ckpt_CDMFusion', self.config.data.dataset + '_stage3')
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename)
                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step, stage)

    def sample_image(self, x_cond,x_cond2, x, stage, canshu = [0,1,1,1,1,0], last=True):
        xs = models.generalized_steps(x, x_cond,x_cond2,canshu,stage, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step, stage):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + '_val_results')
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond2 = x[:, 6:, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x_cond2 = data_transform(x_cond2)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond,x_cond2, x, stage)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)
            x_cond2 = inverse_data_transform(x_cond2)
            for i in range(n):
                utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))