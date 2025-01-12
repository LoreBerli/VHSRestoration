from skvideo import measure
import skvideo.io
import numpy as np
from PIL import Image
vidOURSPATH = "/media/cioni/Seagate/restoration/Thesis-U-nets-for-Video-restoration-and-visual-quality-enhancement-of-old-footage/output/test_best_sarunet/video/test_best_sarunet.mp4"
vidSRUNET = "/media/cioni/Seagate/restoration/Thesis-U-nets-for-Video-restoration-and-visual-quality-enhancement-of-old-footage/output/test_best_sarunet/video/test_best_sarunet.mp4"
vidESRGAN = '/media/cioni/Seagate/restoration/Thesis-U-nets-for-Video-restoration-and-visual-quality-enhancement-of-old-footage/output/test_best_esrgan/video/test_best_esrgan.mp4'

vids = [vidOURSPATH, vidSRUNET, vidESRGAN]
frame_num = 30

vid = "/media/cioni/Seagate/vhs_hd/output/vhstest_final_espcn/video/vhstest_final_espcn.mp4"
dist_video=skvideo.io.vread(vid,as_grey=True,num_frames=frame_num)
#dist_video= np.resize(dist_video,(frame_num,480,638,1))
print(dist_video.shape)
original = "/media/cioni/Seagate/vhs_hd/NEW_TEST/out_405_original.mp4"
original_video = skvideo.io.vread(original,as_grey=True,num_frames=frame_num,outputdict={
                           "-s": f"{dist_video.shape[2]}x{dist_video.shape[1]}",
                       })
print(np.array(original_video[0,:,:,0],dtype=np.uint8).shape)

im0 = Image.fromarray(np.array(original_video[0,:,:,0],dtype=np.uint8))
im1 = Image.fromarray(np.array(dist_video[0,:,:,0],dtype=np.uint8))
im0.show(title="Original")
im1.show(title="distorted")

m=measure.strred(original_video,dist_video)

print(m)
print(np.mean(m[0],0))
# for vid in vids:
#     print(f"Processing {vid}")
#     video = measure.video.psnr(vid, vid)
#     print(video)
