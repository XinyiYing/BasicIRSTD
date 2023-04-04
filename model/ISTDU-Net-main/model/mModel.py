from .ctNet.ctNet import ISTDU_Net

Network = ISTDU_Net

if __name__ == '__main__':
    import torch
    x = torch.ones((1, 1, 512, 512)).to('cuda')
    m = Network().to('cuda')
    m.eval()
    o = m(x)
    print(o)
