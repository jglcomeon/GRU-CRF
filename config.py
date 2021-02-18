import os
import torch
root_path = os.path.abspath(os.path.dirname(__file__))
emb_size = 300
hidden_size = 300

# 通用配置
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')