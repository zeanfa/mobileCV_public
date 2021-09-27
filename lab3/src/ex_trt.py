import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time

# create example data
x1 = torch.ones((1, 3, 224, 224)).cuda()
x2 = torch.zeros((1, 3, 224, 224)).cuda()
x3 = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
# model = alexnet(pretrained=True).eval().cuda()
# model_trt = torch2trt(model, [x1], use_onnx=True)
# torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
# exit()

timest = time.time()
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
print("load time {}".format(time.time()-timest))


timest = time.time()
y_trt1 = model_trt(x1)
print(time.time()-timest)
timest = time.time()
y_trt2 = model_trt(x2)
print(time.time()-timest)
timest = time.time()
y_trt3 = model_trt(x3)
print(time.time()-timest)
