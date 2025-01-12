import time
import numpy as np
from datetime import datetime
import data_loader as dl
import torch
import os
from render import torchToCv2, get_padded_dim
torch.backends.cudnn.benchmark = True
from torch.nn import functional as F
from models import *
import utils
from tqdm import tqdm
import cv2
from pytorch_unet import UNet, SRUnet, SimpleResNet, SARUnet
from rrdbnet import RRDBNet
#import wandb
import torchvision
outpath = "/media/cioni/Seagate/vhs_hd/output/"
def hconcat_resize_min(im_list, interpolation=cv2.INTER_LINEAR):
    im_list_cropped=[im[:,im.shape[1]//4:im.shape[1]-im.shape[1]//4] for im in im_list]
    h_min = max(im.shape[1] for im in im_list_cropped)
    w_max= max(im.shape[0] for im in im_list_cropped)
    # im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation = interpolation)
    #                   for im in im_list_cropped]
    im_list_resize = [cv2.resize(im, (h_min, w_max), interpolation = interpolation)
                      for im in im_list_cropped]
    return cv2.hconcat(src = im_list_resize),im_list_resize[0]

if __name__ == '__main__':
    args = utils.ARArgs()
    enable_write_to_video = False

    #wandb.login()

    print_model = args.VERBOSE
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR

    config = {
              "num_filters": args.N_FILTERS,
              "patch_size": args.PATCH_SIZE,
              "arch": args.ARCHITECTURE,
              "upscale": args.UPSCALE_FACTOR,
              "downsample": args.DOWNSAMPLE,
              "ASPP_DWISE": args.ASPP_DWISE,
        "clip":args.CLIPNAME,
              }
    id_string = args.MODEL_NAME.split("/")[-1]
    id_string = id_string.split(".")[0]+datetime.now().strftime('_%m-%d_%H-%M')

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
        model = SimpleResNet(n_filters=64, n_blocks=6,upscale=dataset_upscale_factor, downsample=args.DOWNSAMPLE)
    elif arch_name == 'sarunet':
        model = SARUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                        downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER,batchnorm=True)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)



    model_path = args.MODEL_NAME
    model.load_state_dict(torch.load(model_path))
    print(model)
    model = model.cuda()
    #if  arch_name in ['srunet', 'sarunet']:
    #   model.reparametrize()

    path = args.CLIPNAME

    reader = torchvision.io.VideoReader(path, "video")
    #reader.seek(80)
    cap = cv2.VideoCapture(path)
    metadata = reader.get_metadata()

    try:

        # creating a folder named data
        if not os.path.exists(outpath+args.OUTPUT_NAME):
            os.makedirs(outpath+args.OUTPUT_NAME)
            os.makedirs(outpath+args.OUTPUT_NAME+"/frames")
            os.makedirs(outpath+args.OUTPUT_NAME+"/video")

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    print(metadata)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_fix, width_fix, padH, padW = get_padded_dim(height, width)
    target_fps = cap.get(cv2.CAP_PROP_FPS)  # cv2.CAP_PROP_FPS get the frame rate of the video
    target_frametime = 1000 / target_fps
    writer = None
    times=[]
    fps_a=[]
    print(frame_count,height,width,height_fix,width_fix,target_fps,target_frametime)
    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(frame_count))
        for i in tqdm_:
            t0 = time.perf_counter()

            cv2_im = next(reader)['data']

            cv2_im_input=F.interpolate(cv2_im.unsqueeze(0).float(),scale_factor=0.8,mode="bicubic")[0]
            #cv2_im_input=F.adaptive_avg_pool2d(cv2_im.float(),[240,320])#cv2_im#F.interpolate(cv2_im.unsqueeze(0).float(),scale_factor=0.5,mode="bicubic")[0]

            cv2_im_input = cv2_im_input.cuda().float()

            x = dl.normalize_img(cv2_im_input / 255.).unsqueeze(0)
            x = x[:,:,0: x.shape[2] - x.shape[2]%8,0:x.shape[3] - x.shape[3]%8]

            #x_orig = dl.normalize_img(cv2_im.float()/ 255.).unsqueeze(0)
            x_orig=cv2_im.numpy()

            x_orig = x_orig[:,0: x_orig.shape[1] - x_orig.shape[1]%16,0:x_orig.shape[2] - x_orig.shape[2]%16]
            x_orig = np.transpose(x_orig, [1, 2, 0])
            #x = F.pad(x, [0, padW, 0, padH])#cv2.resize(x,(720,405),interpolation = cv2.INTER_AREA)#
            #x = F.resize(x, (720, 405), interpolation=cv2.INTER_AREA)
            #
            # out1 = model(x)
            #
            # out = F.interpolate(out1, scale_factor=0.5)
            # out=out*0.98+torch.randn_like(out)*0.02

            #
            out = model(x)
            #out = F.interpolate(out,scale_factor=1.2,mode="bilinear")
            frametime = time.perf_counter() - t0
            fps_a.append([1/frametime])
            times.append([frametime])
            # if frametime < target_frametime * 1e-3:
            #     time.sleep(target_frametime * 1e-3 - frametime)
            out_true = i // (target_fps * 3) % 2 == 0

            #x_orig = torchToCv2(x_orig)
            x_orig = cv2.cvtColor(x_orig, cv2.COLOR_BGR2RGB)
            out = torchToCv2(out)
            splitFrame,crop_or = hconcat_resize_min(im_list=[x_orig, out])

            if i==0:
                print(f"out.shape {out.shape}")
                image_size = (out.shape[1], out.shape[0])
                writer = cv2.VideoWriter(outpath+args.OUTPUT_NAME+"/video/" + args.OUTPUT_NAME + ".mp4",
                                         cv2.VideoWriter_fourcc(*'mp4v'), target_fps, image_size)

                #writer_o = cv2.VideoWriter(outpath+args.OUTPUT_NAME+"/video/" + args.OUTPUT_NAME + "original.mp4",
                #                         cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (crop_or.shape[1], crop_or.shape[0]))

                x_size = (splitFrame.shape[1], splitFrame.shape[0])
                writer2 = cv2.VideoWriter(outpath+args.OUTPUT_NAME+"/video/" + "sideside.mp4",
                                         cv2.VideoWriter_fourcc(*'mp4v'), target_fps, x_size)


            if i%50==0:
                cv2.imwrite(outpath+args.OUTPUT_NAME+"/frames/"f"_{i:03d}.jpg", splitFrame)
            writer.write(image=out)
            #writer_o.write(image=crop_or)
            writer2.write(image=splitFrame)

            tqdm_.set_description("frame time: {}; fps: {}; {}".format(frametime , 1 / frametime, out_true))

    writer.release()
    #writer2.release()
    cap.release()
    mean_fps= np.mean(np.array(fps_a[2:]))
    mean_tims = np.mean(np.array(times[2:]))
    # wandb.log({
    #     "fps": float(mean_fps),
    #     "times": float(mean_tims)
    # })
    print(f"Finish!")