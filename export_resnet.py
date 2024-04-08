import torch,torch.nn as nn,torch.nn.functional as F
import sys
sys.path.append("qdrop/solver")
from recon import compute_hamming_loss
torch.set_grad_enabled(False)
"""
注意所有的权重的值都是要量化的情况下
"""
model = torch.load("resnet50-W8-A8-qdropTrue-202404052357/model.pt")

def exam_quantized(model:torch.nn.Module,img=torch.randn(1,3,224,224)):
    model = model.cuda()
    img = img.cuda()
    ret_dict = dict()
    def get_hook(name):
        def hook(m:nn.Conv2d,i,o):
            x = i[0]
            d = dict()
            d['w_scale'] = m.weight_fake_quant.scale
            d['weight'] = m.weight
            d['qweight'] = m.weight_fake_quant.int_repr(m.weight,True)
            ret_dict[name] = d
        return hook
    hooks = list()
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(get_hook(name)))

    model(img)
    for hook in hooks:
        hook.remove()
    return ret_dict
    


if __name__ == "__main__":
    exam_quantized(model)

