import torch,torch.nn as nn
import numpy as np,matplotlib.pyplot as plt





class HammingLoss(nn.Module):
    def __init__(self,symmetric=True):
        super().__init__()
        self.sym = symmetric
        if symmetric:
            nums = list(range(-128,128,1))
        else:
            nums = list(range(0,256,1))
        hms = [np.binary_repr(i,width=8).replace("0b","").count("1") for i in nums]
        self.register_buffer("hms",torch.tensor(hms).float())

    def forward(self,x,reduce="sum",bit=8):
        if self.hms.device != x.device:
            self.to(x.device)
        if self.sym:
            x.data.clip(-128,127)
            x += 128
        x.data.clip_(0,255)
        low = torch.floor(x).long().clip_(0)
        high = torch.ceil(x).long().clip_(max=255)
        low_val = self.hms[low]
        high_val = self.hms[high]
        frac = x - low.float()
        ret = torch.lerp(low_val,high_val,frac)
        if reduce=="sum":
            return ret.sum()
        else:
            return ret.mean()


if __name__ == "__main__":
    x = torch.rand(1,3,224,224)*127
    ret = HammingLoss()(x)
    print()
