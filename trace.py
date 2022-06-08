import torch
PTH_PATH = "./models/linear_model_27__99.2.pth"
from net import ResNet

model = torch.load(PTH_PATH, map_location=torch.device("cpu"))
model.eval()
example = torch.randn(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")
