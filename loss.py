#%%
import torch
import torch.nn as nn

criterion = nn.MSELoss(reduction='sum')
criterion.cuda()

def get_loss(mode, gt_train, out_train):
    if mode=='mid_sup_mse':
        return mid_sup_mse_loss(gt_train, out_train)
    elif mode=='mse':
        return mse_loss(gt_train, out_train)
    elif mode == 'mse_sum':
        NF, C, H, W = gt_train.size()
        return criterion(gt_train, out_train) / (NF/5*2)
    elif mode == 'l1':
        return torch.mean(torch.abs(gt_train - out_train))
    else:
        raise NotImplementedError

def mse_loss(gt_train, out_train):
    # criterion = nn.MSELoss(reduction='mean')
    # criterion.cuda()
    # loss = criterion(gt_train, out_train)
    loss = torch.mean((gt_train-out_train)**2)
    return loss
    
def mid_sup_mse_loss(gt_train, out_train):
    out_train1, out_train2= out_train
    loss1 = torch.mean((gt_train-out_train1)**2)
    loss2 = torch.mean((gt_train-out_train2)**2)
    loss = (loss1+loss2)/2
    return loss
#%%
def test_torch_MSELoss():
    input = torch.randn(1, 2, requires_grad=True)
    target = torch.randn(1, 2)
    manual = torch.mean((input-target)**2)
    print(input)
    print(target)
    print(manual)
    loss = nn.MSELoss(reduction="mean")
    output = loss(input, target)
    print(output)
    loss = nn.MSELoss(reduction="sum")
    output = loss(input, target)
    print(output)
    # %%
    print('test image')
    input = torch.randn(4, 3, 960, 540, requires_grad=True)
    target = torch.randn(4, 3, 960, 540,)
    manual = torch.mean((input-target)**2)
    # 2*4*3*960*540
    # print(input)
    # print(target)
    print(manual)
    loss = nn.MSELoss(reduction="mean")
    output = loss(input, target)
    print(output)
    loss = nn.MSELoss(reduction="sum") # 2*4*3*960*540
    output = loss(input, target)
    print(output)
# %%
