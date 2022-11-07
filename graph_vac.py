import matplotlib.pyplot as plt
import numpy as np

# VMAF
#c1 =[[9.20,20248],[15.686,10099],[32.649,3394]]
#c2 =[[14.45,20248],[20.09,10099],[37.16,3394]]
#c3 =[[2.78,94492],[6.48,40906],[16.16,12864]]
crfs = [18, 21, 23, 30]
bitrates = [20248, 13444, 10099, 3394]
bitrates_1080 = [94492, 57040, 40906, 18028, 12864]
vamf_rec____ = [90.822463, 87.354537, 84.314058, 67.351301]
vamf_vac = [88.345628, 84.991111, 82.044096, 65.663400]
vamf_1080 = [97.227430, 95.261442, 93.510012, 86.881566, 83.047513]
lpips_ = [0.157341, 0.168993, 0.179580, 0.220343, 0.243500]
lpips_vac = [0.181244, 0.185965, 0.191463, 0.236555]
lpips_encoded_1080 = [0.076712, 0.103519, 0.119601, 0.160785, 0.180537]

c1 = list(zip(vamf_rec____, bitrates))
c2 = list(zip(vamf_vac, bitrates))
c3 = list(zip(vamf_1080, bitrates_1080))
c1p, c2p, c3p = np.array(c1), np.array(c2), np.array(c3)
crf = [18, 21, 23, 30]
cs = [c1, c2]
plt.plot(100-c1p[:, 0], c1p[:, 1], marker=".",
         label=("ours 540 + SR"), color="orange")
plt.plot(100-c2p[:, 0], c2p[:, 1], marker=".",
         label=("VAC 540 + SR"), color="cyan")
# plt.plot(100-c3p[:,0],c3p[:,1],marker=".",label=("1080"),color="green")
plt.xlabel("Distortion (100-VMAF)")
plt.ylabel("Bitrate - kbps")
for i, c in enumerate(cs):

    for j, p in enumerate(c):
        print(crf[j])
        plt.annotate(crf[j], (100-p[0], p[1]))

plt.title("Figure 4: Rate/VMAF-distortion curve the varying of the compression CRF.")
plt.legend()
plt.show()
# LPIPS

c1 = list(zip(lpips_, bitrates,))
c2 = list(zip(lpips_vac, bitrates))
c3 = list(zip(lpips_encoded_1080, bitrates_1080))

c1p, c2p, c3p = np.array(c1), np.array(c2), np.array(c3)

cs = [c1, c2]
plt.plot(c1p[:, 0], c1p[:, 1], marker=".",
         label=("ours 540 + SR"), color="orange")
plt.plot(c2p[:, 0], c2p[:, 1], marker=".",
         label=("VAC 540 + SR"), color="cyan")
# plt.plot(c3p[:,0],c3p[:,1],marker=".",label=("1080"),color="green")
plt.xlabel("Distortion LPIPS")
plt.ylabel("Bitrate - kbps")
for i, c in enumerate(cs):

    for j, p in enumerate(c):
        print(crf[j])
        plt.annotate(crf[j], (p[0], p[1]))

plt.title("Figure 4: Rate/LPIPS-distortion curve the varying of the compression CRF.")
plt.legend()
plt.show()
