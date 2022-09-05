import matplotlib.pyplot as plt
import numpy as np

#################VMAF
c1 =[[9.20,20248],[15.686,10099],[32.649,3394]]
c2 =[[14.45,20248],[20.09,10099],[37.16,3394]]
c3 =[[2.78,94492],[6.48,40906],[16.16,12864]]

c1p,c2p,c3p = np.array(c1),np.array(c2),np.array(c3)
crf=[18,23,30]
cs=[c1,c2,c3]
plt.plot(c1p[:,0],c1p[:,1],marker=".",label=("540 + SR"),color="orange")
plt.plot(c2p[:,0],c2p[:,1],marker=".",label=("540"),color="cyan")
plt.plot(c3p[:,0],c3p[:,1],marker=".",label=("1080"),color="green")
plt.xlabel("Distortion (100-VMAF)")
plt.ylabel("Bitrate - kbps")
for i,c in enumerate(cs):

    for j,p in enumerate(c):
        print(crf[j])
        plt.annotate(crf[j],(p[0],p[1]))

plt.title("Figure 4: Rate/VMAF-distortion curve the varying of the compression CRF.")
plt.legend()
plt.show()
######################LPIPS
c1 =[[0.1573,20248],[0.1795,10099],[0.2435,3394]]
c2 =[[0.2340,20248],[0.2621,10099],[0.3253,3394]]
c3 =[[0.0767,94492],[0.1196,40906],[0.1805,12864]]

c1p,c2p,c3p = np.array(c1),np.array(c2),np.array(c3)
crf=[18,23,30]
cs=[c1,c2,c3]
plt.plot(c1p[:,0],c1p[:,1],marker=".",label=("540 + SR"),color="orange")
plt.plot(c2p[:,0],c2p[:,1],marker=".",label=("540"),color="cyan")
plt.plot(c3p[:,0],c3p[:,1],marker=".",label=("1080"),color="green")
plt.xlabel("Distortion LPIPS")
plt.ylabel("Bitrate - kbps")
for i,c in enumerate(cs):

    for j,p in enumerate(c):
        print(crf[j])
        plt.annotate(crf[j],(p[0],p[1]))

plt.title("Figure 4: Rate/LPIPS-distortion curve the varying of the compression CRF.")
plt.legend()
plt.show()
