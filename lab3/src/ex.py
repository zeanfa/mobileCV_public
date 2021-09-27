import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time

# create some regular pytorch model...
timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("load time {}".format(time.time()-timest))

# create example data
x1 = torch.ones((1, 3, 224, 224)).cuda()
x2 = torch.ones((1, 3, 224, 224)).cuda()
x3 = torch.ones((1, 3, 224, 224)).cuda()

timest = time.time()
y1 = model(x1)
print(time.time()-timest)

timest = time.time()
y2 = model(x2)
print(time.time()-timest)

timest = time.time()
y3 = model(x3)
print(time.time()-timest)
#print("y {}".format(y))
