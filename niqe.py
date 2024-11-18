import pyiqa
import torch
import os
# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid_metric = pyiqa.create_metric('niqe')
# fid_metric = pyiqa.create_metric('brisque')


score = 0
n = 0
for root,_,files in os.walk('result'):
    for file in files:
        score += fid_metric(os.path.join(root,file)).item()
        n += 1
    
print(score/n)

