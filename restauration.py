import time
import data_loader as dl
import torch
import os
from render import torchToCv2, get_padded_dim
torch.backends.cudnn.benchmark = True
from models import *
import utils
from tqdm import tqdm
import cv2
from pytorch_unet import UNet, SRUnet, SimpleResNet, SARUnet


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation = interpolation)
                      for im in im_list]
    return cv2.hconcat(src = im_list_resize)

if __name__ == '__main__':
    args = utils.ARArgs()
    enable_write_to_video = False
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR

    if arch_name == 'srunet':
        model = SRUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                       downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    elif arch_name == 'unet':
        model = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS)
    elif arch_name == 'srgan':
        model = SRResNet()
    elif arch_name == 'espcn':
        model = SimpleResNet(n_filters=64, n_blocks=6)
    elif arch_name == 'sarunet':
        model = SARUnet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
                        downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    model_path = args.MODEL_NAME
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    if  arch_name in ['srunet', 'sarunet']:
       model.reparametrize()

    path = args.CLIPNAME
    cap = cv2.VideoCapture(path)
    reader = torchvision.io.VideoReader(path, "video")
    metadata = reader.get_metadata()

    try:

        # creating a folder named data
        if not os.path.exists("output/"+args.OUTPUT_NAME):
            os.makedirs("output/"+args.OUTPUT_NAME)
            os.makedirs("output/"+args.OUTPUT_NAME+"/frames")
            os.makedirs("output/"+args.OUTPUT_NAME+"/video")

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_fix, width_fix, padH, padW = get_padded_dim(height, width)
    target_fps = cap.get(cv2.CAP_PROP_FPS)  # cv2.CAP_PROP_FPS get the frame rate of the video
    target_frametime = 1000 / target_fps
    writer = None

    model = model.eval()
    with torch.no_grad():
        tqdm_ = tqdm(range(frame_count))
        for i in tqdm_:
            t0 = time.time()

            cv2_im = next(reader)['data']
            cv2_im = cv2_im.cuda().float()

            x = dl.normalize_img(cv2_im / 255.).unsqueeze(0)
            x = F.pad(x, [0, padW, 0, padH])
            out = model(x)

            frametime = time.time() - t0
            if frametime < target_frametime * 1e-3:
                time.sleep(target_frametime * 1e-3 - frametime)
            out_true = i // (target_fps * 3) % 2 == 0

            x = torchToCv2(x)
            out = torchToCv2(out)
            splitFrame = hconcat_resize_min(im_list=[x, out])

            if i==0:
                print(f"out.shape {out.shape}")
                image_size = (splitFrame.shape[1], splitFrame.shape[0])
                writer = cv2.VideoWriter("output/"+args.OUTPUT_NAME+"/video/" + args.OUTPUT_NAME + ".mp4",
                                         cv2.VideoWriter_fourcc(*'mp4v'), 60, image_size)

            if i%500==0:
                cv2.imwrite("output/"+args.OUTPUT_NAME+"/frames/"f"_{i:03d}.jpg", splitFrame)
            writer.write(image=splitFrame)

            tqdm_.set_description("frame time: {}; fps: {}; {}".format(frametime * 1e3, 1000 / frametime, out_true))

    writer.release()
    cap.release()
    print(f"Finish!")