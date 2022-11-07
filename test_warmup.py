import torch
from WarmupScheduler import GradualWarmupScheduler

model = torch.nn.Sequential(torch.nn.Linear(32,32))

opt = torch.optim.Adam(lr=1e-06, params=model.parameters())
sched_c = GradualWarmupScheduler(opt,101,200,after_scheduler=torch.optim.lr_scheduler.LinearLR(opt,1.0,0.1,total_iters=500))

for i in range(0,800):
    if(i%20==0):
        print(i,sched_c.get_lr(),sched_c.get_last_lr())
    sched_c.step(i)
