import os, utils
import time

args = utils.ARArgs()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE
import math
import numpy as np
import data_loader as dl
import torch
from torch.nn import functional as F
from torch import nn as nn
from torch.utils.data import DataLoader
#import pytorch_ssim  # courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import tqdm
import lpips  # courtesy of https://github.com/richzhang/PerceptualSimilarity
from models import Discriminator, DiscriminatorESRGAN, \
    SRResNet  # courtesy of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
from pytorch_unet import SRUnet, UNet, SimpleResNet, SARUnet,SARUnet_np,SARUnet_noASP
from rrdbnet import RRDBNet
from WarmupScheduler import GradualWarmupScheduler
import wandb
from datetime import datetime
from torchvision import transforms

if __name__ == '__main__':
    args = utils.ARArgs()
    torch.autograd.set_detect_anomaly(True)

    wandb.login()
    finetuning=False
    print_model = args.VERBOSE
    arch_name = args.ARCHITECTURE

    dataset_upscale_factor = args.UPSCALE_FACTOR
    n_epochs = args.N_EPOCHS

    config = {"batch_size": args.BATCH_SIZE,  # fisso
              "learning_rate": args.LR,  # fisso
              "num_epochs": args.N_EPOCHS,  # da terminale
              "num_filters": args.N_FILTERS,
              "patch_size": args.PATCH_SIZE,
              "arch": args.ARCHITECTURE,
              "upscale": args.UPSCALE_FACTOR,
              "downsample": args.DOWNSAMPLE,
              "ASPP_DWISE": args.ASPP_DWISE
              }
    id_string = args.ARCHITECTURE + "_nf_" + str(args.N_FILTERS) + datetime.now().strftime('_%m-%d_%H-%M') + "_" + str(
        args.PATCH_SIZE) + "_" + args.RES
    wandb.config = config
    mode = "disabled" if args.DEBUG else None
    wandb.init(project='SuperRes', config=config, id=id_string, mode=mode,
               entity="cioni")

    if arch_name == 'srunet':
        model = SRUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'unet':
        model = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,downsample=args.DOWNSAMPLE)
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'esrgan':
        model = RRDBNet(3, 3, scale=dataset_upscale_factor, nf=64, nb=23, downsample=args.DOWNSAMPLE)

    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6,upscale=dataset_upscale_factor,downsample=args.DOWNSAMPLE)
    elif arch_name == 'sarunet':
        model = SARUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                        downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER,batchnorm=True)
    elif arch_name == 'sarunet_noplus':
        model = SARUnet_np(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                        downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'sarunet_ASPP':
        model = SARUnet_noASP(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                        downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)

    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    if args.MODEL_NAME is not None:
        print("Loading model: ", args.MODEL_NAME)
        state_dict = torch.load(args.MODEL_NAME)
        model.load_state_dict(state_dict)
    print(arch_name)
    print(model)
    model.down_factor=args.DOWNSAMPLE
    os.mkdir(args.EXPORT_DIR + "/" + id_string)
    wandb.watch(model)
    usa_amp=False
    if finetuning:
        for m in model.state_dict():
            if not m.startswith('conv_last'):
                model.state_dict()[m].requires_grad = False
    critic = Discriminator()
    model = model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.save(model.state_dict(),
               args.EXPORT_DIR + "/" + id_string + '/' + 'last.pkl')
    discriminator_optimizer = torch.optim.Adam(lr=0.00004, params=critic.parameters())
    generator_optimizer = torch.optim.Adam(lr=0.00004, params=model.parameters())

    sched_c = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, 60 * 1000, gamma=0.5)
    sched_g = torch.optim.lr_scheduler.StepLR(generator_optimizer, 60 * 1000, gamma=0.5)
    #sched_c = GradualWarmupScheduler(discriminator_optimizer,130,6000,after_scheduler=torch.optim.lr_scheduler.LinearLR(discriminator_optimizer,1.0,0.02,total_iters=8000))#torch.optim.lr_scheduler.StepLR(discriminator_optimizer, 205, gamma=0.8)
    #sched_g = GradualWarmupScheduler(generator_optimizer,130,6000,after_scheduler=torch.optim.lr_scheduler.LinearLR(generator_optimizer,1.0,0.02,total_iters=8000))#torch.optim.lr_scheduler.ChainedScheduler([torch.optim.lr_scheduler.LinearLR(generator_optimizer,)])#torch.optim.lr_scheduler.StepLR(generator_optimizer, 205, gamma=0.8)
    lpips_loss = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex = lpips_loss#lpips.LPIPS(net='alex', version='0.1')
    ssim = SSIM(data_range=2, size_average=True, channel=3)#pytorch_ssim.SSIM()
    scaler = torch.cuda.amp.GradScaler()
    scalerg = torch.cuda.amp.GradScaler()
    model.to(device)
    model.train()
    lpips_loss.to(device)
    lpips_alex.to(device)
    critic.to(device)
    ssim.to(device)

    dataset_train = dl.ARDataLoader2(hq_path=args.HQ_DATASET_DIR,lq_path=args.LQ_DATASET_DIR, patch_size=args.PATCH_SIZE, eval=False, use_ar=True,
                                     res=str(args.RES), set=args.set, dataset_upscale_factor=int(args.UPSCALE_FACTOR),
                                     rescale_factor=args.DOWNSAMPLE,seed=args.SEED)

    dataset_test = dl.ARDataLoader2(hq_path=args.HQ_DATASET_DIR,lq_path=args.LQ_DATASET_DIR, patch_size=args.PATCH_SIZE, eval=True, use_ar=True,
                                    res=str(args.RES), set=args.set, dataset_upscale_factor=int(args.UPSCALE_FACTOR),
                                    rescale_factor=args.DOWNSAMPLE,seed=args.SEED)

    data_loader = DataLoader(dataset=dataset_train, batch_size=args.BATCH_SIZE, num_workers=8, shuffle=True,
                             pin_memory=True)

    data_loader_eval = DataLoader(dataset=dataset_test, batch_size=args.BATCH_SIZE, num_workers=1, shuffle=False,
                                  pin_memory=True)


    loss_discriminator = nn.BCEWithLogitsLoss()

    print(f"Total epochs: {n_epochs}; Steps per epoch: {len(data_loader)}")

    # setting loss weights
    w0, w1, l0 = args.W0, args.W1, args.L0
    wandb.watch(model, criterion=[loss_discriminator, lpips_loss, lpips_alex], log='all')
    #l0 = 0.0001
    display_it = iter(data_loader_eval)
    for e in range(n_epochs):
        display_it = iter(data_loader_eval)
        # if e == max(n_epochs - starting_epoch, 0):
        #     utils.adjust_learning_rate(discriminator_optimizer, 0.1)
        #     utils.adjust_learning_rate(generator_optimizer, 0.1)
        loss_discr = 0.0

        loss_gen = 0.0
        loss_bce_gen = 0.0
        lpips_list=[]
        ssim_list=[]
        gloss_list=[]
        dloss_list=[]
        print("Epoch:", e)

        tqdm_ = tqdm.tqdm(data_loader)
        step = 0

        for batch in tqdm_:
            # if e==0:
            #     l0=min(args.L0,l0+0.000005)
            # else:
            #     l0=args.L0
            model.train()
            critic.train()
            discriminator_optimizer.zero_grad()
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)
            batch_dim = x.shape[0]
            y_fake = model(x)

        # train critic phase
            pred_true = critic(y_true)

        # forward pass on true
            loss_true = torch.mean((pred_true-1.0)**2)  # loss_discriminator(pred_true, torch.ones_like(pred_true))

        # then updates on fakes
            pred_fake = critic(y_fake.detach())
            loss_fake = torch.mean((pred_fake)**2)  # loss_discriminator(pred_fake, torch.zeros_like(pred_fake))
            loss_discr = loss_true + loss_fake
            loss_discr *= 0.5


            dloss_list.append(loss_discr.detach().cpu().numpy())
            # scaler.scale(loss_discr).backward()
            # scaler.step(discriminator_optimizer)
            # scaler.update()
            loss_discr.backward()
            discriminator_optimizer.step()

            loss_discr = float(loss_discr)

            ## train generator phase
            generator_optimizer.zero_grad()
            lpips_loss_ = lpips_loss(y_fake, y_true).mean()
            #with torch.autocast(device_type='cuda', dtype=torch.float16,enabled=usa_amp):
            ssim_loss = 1.0 - ssim(1.0 + y_fake, 1.0 + y_true)
            pred_fake = critic(y_fake)
            bce = torch.mean((pred_fake-1.0)**2)  # loss_discriminator(pred_fake, torch.ones_like(pred_fake)) #propobabilità dei falsi di essere scambiati per veri
            loss_gen = w0 * lpips_loss_ + w1 * ssim_loss + l0 * bce  # loss proposta da minimizzare (però nel paper ssim
            loss_bce_gen=l0 * bce

            lpips_list.append(lpips_loss_.detach().cpu().numpy())
            ssim_list.append(ssim_loss.detach().cpu().numpy())
            gloss_list.append(loss_gen.detach().cpu().numpy())
            # scalerg.scale(loss_gen).backward()
            # scalerg.step(generator_optimizer)
            # scalerg.update()
            loss_gen.backward()  # retropagazione degli errori per aggiornare i pesi del generatore secondo la loss proposta
            generator_optimizer.step()  # aggiornamenti dei gradienti

            tqdm_.set_description(
                'L D: {:.5f}; L C: {:.5f}; BCE/L0: {:.5f}'.format(loss_discr,
                                                                         float(loss_gen) - float(
                                                                             l0 * loss_bce_gen),
                                                                     float(loss_bce_gen)))
            if step % 300==0:
                torch.save(model.state_dict(),
                           args.EXPORT_DIR + "/" + id_string + '/' + 'last.pkl')

            if step % 20 == 0:
                lpips_mean_list=np.mean(lpips_list)
                ssim_mean_list = np.mean(ssim_list)
                gloss_mean_list =np.mean(gloss_list)
                dloss_mean_list = np.mean(dloss_list)
                # Log your Table to W&B
                wandb.log({"Loss discriminante - Training-set": dloss_mean_list,
                           "Loss generatore - Training-set": gloss_mean_list,
                           "Loss lpips (vgg-net) - Training-set": lpips_mean_list,
                           "Loss ssim - Training-set": ssim_mean_list,
                           "Content loss - Training-set": float(loss_gen),
                           "Epoch": e,
                           "LR_g" : sched_g.get_lr()[0],
                           "LR_c": sched_c.get_lr()[0],
                           "l0":bce
                           },
                          )
                # Re-set the empty lists
                lpips_list = []
                ssim_list = []
                gloss_list=[]
                dloss_list=[]


            if step % 300 == 0:
                model.eval()
                with torch.no_grad():
                    x, y_true = next(display_it)
                    x = x.to(device)
                    y_true = y_true.to(device)
                    y_fake = model(x)
                model.train()
                wandb.log({"Image_LR": [wandb.Image(im) for im in x],
                           "Image_fast-sr-unet": [wandb.Image(im) for im in torch.clamp(y_fake,-1.0,1.0)],
                           "Image_HQ": [wandb.Image(im) for im in y_true]})
            step += 1
            sched_c.step()
            sched_g.step()

        if (e + 1) % args.VALIDATION_FREQ == 0:
            print("Validation phase")

            ssim_validation = []
            lpips_validation = []

            tqdm_ = tqdm.tqdm(data_loader_eval)
            model.eval()
            for batch in tqdm_:
                x, y_true = batch
                with torch.no_grad():
                    x = x.to(device)
                    y_true = y_true.to(device)
                    y_fake = model(x)
                    ssim_val = ssim(1.0+y_fake, 1.0+y_true).mean()
                    lpips_val = lpips_alex(y_fake, y_true).mean()
                    ssim_validation += [float(ssim_val)]
                    lpips_validation += [float(lpips_val)]

            ssim_mean = np.array(ssim_validation).mean()
            lpips_mean = np.array(lpips_validation).mean()

            print(f"Val SSIM: {ssim_mean}, Val LPIPS: {lpips_mean}")
            wandb.log({
                "Loss SSIM - validation set": ssim_mean,
                "Loss LPIPS (alex-net) - validation set": lpips_mean
            })

            torch.save(model.state_dict(),
                       args.EXPORT_DIR + "/" + id_string + '/' + '{0}_epoch{1}_ssim{2:.4f}_lpips{3:.4f}_res{4}.pkl'.format(
                           arch_name, e, ssim_mean, lpips_mean,
                           args.RES))