#%%

import torch

model = torch.nn.Linear(1024,1024, bias=False).cuda() 
print(torch.cuda.memory_allocated())

#%%