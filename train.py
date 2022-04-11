import os, utils
import time

args = utils.ARArgs()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

import numpy as np
import data_loader as dl
import torch
from torch.nn import functional as F
from torch import nn as nn
from torch.utils.data import DataLoader
import pytorch_ssim  # courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
import tqdm
import lpips  # courtesy of https://github.com/richzhang/PerceptualSimilarity
from models import Discriminator, DiscriminatorESRGAN,\
    SRResNet  # courtesy of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
from pytorch_unet import SRUnet, UNet, SimpleResNet, SARUnet
from rrdbnet import RRDBNet

import wandb
from datetime import datetime
from torchvision import transforms



if __name__ == '__main__':
    args = utils.ARArgs()
    torch.autograd.set_detect_anomaly(True)

    wandb.login()

    print_model = args.VERBOSE
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR
    n_epochs = args.N_EPOCHS


    config = {"batch_size": args.BATCH_SIZE, #fisso
             "learning_rate": args.LR, #fisso
              "num_epochs": args.N_EPOCHS, #da terminale
              "num_filters": args.N_FILTERS,
              "patch_size": args.PATCH_SIZE
              }
    id_string = args.ARCHITECTURE+"nf_"+str(args.N_FILTERS)+"_"+datetime.now().strftime('_%m-%d_%H-%M')+"_"+str(args.PATCH_SIZE)+"_"+args.RES
    wandb.config = config
    mode = "disabled" if args.DEBUG else None
    wandb.init(project='SuperRes', config=config, id=id_string,mode=mode,
               entity="cioni")

    if arch_name == 'srunet':
        model = SRUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'unet':
        model = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS)
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'esrgan':
        model = RRDBNet(3,3,scale=dataset_upscale_factor,nf=64,nb=6,downsample=args.DOWNSAMPLE)

    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6)
    elif arch_name == 'sarunet':
        model = SARUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    if args.MODEL_NAME is not None:
        print("Loading model: ", args.MODEL_NAME)
        state_dict = torch.load(args.MODEL_NAME)
        model.load_state_dict(state_dict)

    print(model)

    wandb.watch(model)

    critic = DiscriminatorESRGAN()
    model = model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    critic_opt = torch.optim.Adam(lr=args.LR/2.0, params=critic.parameters())
    gan_opt = torch.optim.Adam(lr=args.LR, params=model.parameters())
    sched_c=torch.optim.lr_scheduler.StepLR(critic_opt,205*10,gamma=0.5)
    sched_g = torch.optim.lr_scheduler.StepLR(gan_opt, 205 * 10, gamma=0.5)
    lpips_loss = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex = lpips.LPIPS(net='alex', version='0.1')
    ssim = pytorch_ssim.SSIM()

    model.to(device)
    lpips_loss.to(device)
    lpips_alex.to(device)
    critic.to(device)

    dataset_train = dl.ARDataLoader2(path=str(args.DATASET_DIR), patch_size=args.PATCH_SIZE, eval=False, use_ar=True,
                                     res=str(args.RES), set=args.set, dataset_upscale_factor=int(args.UPSCALE_FACTOR),rescale_factor=args.DOWNSAMPLE)
    dataset_test = dl.ARDataLoader2(path=str(args.DATASET_DIR), patch_size=args.PATCH_SIZE, eval=True, use_ar=True,
                                    res=str(args.RES), set=args.set, dataset_upscale_factor=int(args.UPSCALE_FACTOR),rescale_factor=args.DOWNSAMPLE)

    data_loader = DataLoader(dataset=dataset_train, batch_size=args.BATCH_SIZE, num_workers=4, shuffle=True,
                             pin_memory=True)
    data_loader_eval = DataLoader(dataset=dataset_test, batch_size=args.BATCH_SIZE, num_workers=4, shuffle=True,
                                  pin_memory=True)

    loss_discriminator = nn.BCEWithLogitsLoss()

    print(f"Total epochs: {n_epochs}; Steps per epoch: {len(data_loader)}")

    # setting loss weights
    w0, w1, l0 = args.W0, args.W1, args.L0


    wandb.watch(model, criterion= [loss_discriminator, lpips_loss, lpips_alex], log='all')



    for e in range(n_epochs):

        # if e == max(n_epochs - starting_epoch, 0):
        #     utils.adjust_learning_rate(critic_opt, 0.1)
        #     utils.adjust_learning_rate(gan_opt, 0.1)

        loss_discr = 0.0
        loss_gen = 0.0
        loss_bce_gen = 0.0

        print("Epoch:", e)

        tqdm_ = tqdm.tqdm(data_loader)
        step = 0
        for batch in tqdm_:
            #if e ==0:
            #    l0=0.00001 if step <1000 else min(args.L0,l0+0.00001)
            #else:
            l0=args.L0
            model.train()
            critic.train()
            critic_opt.zero_grad()

            x, y_true = batch



            x = x.to(device)
            y_true = y_true.to(device)

            y_fake = model(x)


            # train critic phase
            batch_dim = x.shape[0]

            pred_true = critic(y_true)

            # forward pass on true
            loss_true = torch.mean(F.relu(1.0-pred_true))#loss_discriminator(pred_true, torch.ones_like(pred_true))

            # then updates on fakes
            pred_fake = critic(y_fake.detach())
            loss_fake = torch.mean(F.relu(1.0+pred_fake))#loss_discriminator(pred_fake, torch.zeros_like(pred_fake))

            loss_discr = loss_true + loss_fake
            loss_discr *= 0.5

            loss_discr.backward()
            critic_opt.step()

            loss_discr = float(loss_discr)

            ## train generator phase
            gan_opt.zero_grad()


            lpips_loss_ = lpips_loss(y_fake, y_true).mean()
            ssim_loss = 1.0 - ssim(y_fake, y_true)
            pred_fake = critic(y_fake)
            bce = torch.mean(-pred_fake)#loss_discriminator(pred_fake, torch.ones_like(pred_fake)) #propobabilità dei falsi di essere scambiati per veri
            loss_gen = w0 * lpips_loss_ + w1 * ssim_loss + l0 * bce #loss proposta da minimizzare (però nel paper ssim

            loss_gen.backward() #retropagazione degli errori per aggiornare i pesi del generatore secondo la loss proposta
            gan_opt.step() #aggiornamenti dei gradienti

            tqdm_.set_description(
                'L discr: {:.5f}; C loss: {:.5f}; BCE/L0: {:.5f}'.format(loss_discr,
                                                                                  float(loss_gen) - float(
                                                                                      l0 * loss_bce_gen),
                                                                                  float(loss_bce_gen)))

            wandb.log({"Loss discriminante - Training-set":loss_discr,
                      "Loss generatore - Training-set": loss_gen,
                       "Loss lpips (vgg-net) - Training-set": lpips_loss_,
                      "Loss ssim - Training-set": ssim_loss,
                       "Content loss - Training-set": float(loss_gen) - float(l0 * loss_bce_gen),
                       "Epoch":e
                       },
                      )
            # Log your Table to W&B


            if step%100==0:
                wandb.log({"Image_LR": [wandb.Image(im) for im in x],
                           "Image_fast-sr-unet": [wandb.Image(im) for im in y_fake],
                           "Image_HQ": [wandb.Image(im) for im in y_true]})
            step += 1

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
                    ssim_val = ssim(y_fake, y_true).mean()
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
                       args.EXPORT_DIR+'/'+'{0}_epoch{1}_ssim{2:.4f}_lpips{3:.4f}_res{4}.pkl'.format(arch_name, e, ssim_mean, lpips_mean,
                                                                                 args.RES))
            sched_c.step()
            sched_g.step()
            # having critic's weights saved was not useful, better sparing storage!
            # torch.save(critic.state_dict(), 'critic_gan_{}.pkl'.format(e + starting_epoch))



