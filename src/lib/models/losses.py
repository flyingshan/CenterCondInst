# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F


def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

def dice_loss(input, target):
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))


class DiceLoss(nn.Module):
    def __init__(self,feat_channel):
        super(DiceLoss, self).__init__()
        self.feat_channel=feat_channel

    def forward(self, seg_feat, conv_weight, mask, ind, target):

        mask_loss=0.
        batch_size = seg_feat.size(0)
        weight = _tranpose_and_gather_feat(conv_weight, ind)
        h,w = seg_feat.size(-2),seg_feat.size(-1)
        x,y = ind%w,ind/w
        x_range = torch.arange(w).float().to(device=seg_feat.device)
        y_range = torch.arange(h).float().to(device=seg_feat.device)
        # to replace torch.meshgrid
        # a = torch.linspace(0, w-1, w)
        # b = torch.linspace(0, h-1, h)
        # x_grid = a.repeat(h, 1).cuda()
        # y_grid = b.repeat(w, 1).cuda()
        # end
        y_grid, x_grid = torch.meshgrid([y_range, x_range])
        # torch.Size([4, 169, 128, 128]) torch.Size([4, 128]) torch.Size([4, 128, 169])
        # print(conv_weight.size(), ind.size(), weight.size())
        # print(seg_feat.size())[4, 8, 128, 128])
        # print(mask.size()) ([4, 128])
        for i in range(batch_size):
            num_obj = target[i].size(0)
            # torch.Size([N, n_inst, 128, 128]) target.size()
            # target是生成的gt，也就是target mask

            conv1w,conv1b,conv2w,conv2b,conv3w,conv3b= \
                torch.split(weight[i,:num_obj],[(self.feat_channel+2)*self.feat_channel,self.feat_channel,
                                          self.feat_channel**2,self.feat_channel,
                                          self.feat_channel,1],dim=-1)
            y_rel_coord = (y_grid[None,None] - y[i,:num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
            x_rel_coord = (x_grid[None,None] - x[i,:num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
            
            feat = seg_feat[i][None].repeat([num_obj,1,1,1])
            feat = torch.cat([feat,x_rel_coord, y_rel_coord],dim=1).view(1,-1,h,w)

            conv1w=conv1w.contiguous().view(-1,self.feat_channel+2,1,1)
            conv1b=conv1b.contiguous().flatten() # .flatten() -> .view(-1)
            feat = F.conv2d(feat,conv1w,conv1b,groups=num_obj).relu()

            conv2w=conv2w.contiguous().view(-1,self.feat_channel,1,1)
            conv2b=conv2b.contiguous().flatten() #
            feat = F.conv2d(feat,conv2w,conv2b,groups=num_obj).relu()

            conv3w=conv3w.contiguous().view(-1,self.feat_channel,1,1)
            conv3b=conv3b.contiguous().flatten() #
            feat = F.conv2d(feat,conv3w,conv3b,groups=num_obj).sigmoid().squeeze()

            true_mask = mask[i,:num_obj,None,None].float()
            mask_loss+=dice_loss(feat*true_mask,target[i]*true_mask)

        return mask_loss/batch_size

class FastDiceLoss(nn.Module):
    def __init__(self, feat_channel):
        super(FastDiceLoss, self).__init__()
        self.feat_channel = feat_channel

    def forward(self, seg_feat, conv_weight, mask, ind, target, nums):
        """
        seg_feat: (N, 8, 128, 128) 应该是mask_branch输出的特征图
        conv_weight: (N, 169, h, w)即为controller输出的特征图
        mask: (N, max_obj) 其中有目标的为1，非目标的为0
        ind: (N, max_obj) max_obj由 dataset/coco.py下的max_obj给出
           含义是对应中心点的序号（也可以说是坐标信息）
        target: (N, max_per_n_inst, h, w)为生成的gt mask
            （类似于语义分割的gt，每张图片每个实例都有一个h*w的gt）
        others:
        # weight: size(N, max_obj, 169)应该是把每个ind指示的中心点处（只有前n_inst处ind有值，其它为0）的weight取出来
        """

        # end
        #         y_grid, x_grid = torch.meshgrid([y_range, x_range])

        def produce_mask_head_inputs(seg_feat, nums, x_grid, y_grid, x, y, batch_size):
            num_obj = nums[0].item()
            y_rel_coord = (y_grid[None, None] - y[0, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(
                -1).float()) / 128.
            x_rel_coord = (x_grid[None, None] - x[0, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(
                -1).float()) / 128.
            feat = seg_feat[0][None].repeat([num_obj, 1, 1, 1])
            feat = torch.cat([feat, x_rel_coord, y_rel_coord], dim=1).view(1, -1, h, w)

            for i in range(1, batch_size):
                num_obj = nums[i].item()
                y_rel_coord = (y_grid[None, None] - y[i, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(
                    -1).float()) / 128.
                x_rel_coord = (x_grid[None, None] - x[i, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(
                    -1).float()) / 128.

                feat_temp = seg_feat[i][None].repeat([num_obj, 1, 1, 1])
                feat_temp = torch.cat([feat_temp, x_rel_coord, y_rel_coord], dim=1).view(1, -1, h, w)
                feat = torch.cat([feat, feat_temp], dim=1)
            return feat

        def parse_dynamic_params(params, channels, weight_nums, bias_nums):
            assert params.dim() == 2
            assert len(weight_nums) == len(bias_nums)
            assert params.size(1) == sum(weight_nums) + sum(bias_nums)

            num_insts = params.size(0)
            num_layers = len(weight_nums)
            """
            in size: (10, 169)
            out size: 
            torch.Size([10, 80])
            torch.Size([10, 64])
            torch.Size([10, 8])
            torch.Size([10, 8])
            torch.Size([10, 8])
            torch.Size([10, 1])
            """
            params_splits = list(torch.split_with_sizes(
                params, weight_nums + bias_nums, dim=1
            ))

            weight_splits = params_splits[:num_layers]
            bias_splits = params_splits[num_layers:]

            for l in range(num_layers):
                if l < num_layers - 1:
                    # out_channels x in_channels x 1 x 1
                    weight_splits[l] = weight_splits[l].contiguous().view(num_insts * channels, -1, 1, 1)
                    bias_splits[l] = bias_splits[l].contiguous().view(num_insts * channels)
                else:
                    # out_channels x in_channels x 1 x 1
                    weight_splits[l] = weight_splits[l].contiguous().view(num_insts * 1, -1, 1, 1)
                    bias_splits[l] = bias_splits[l].contiguous().view(num_insts)
            """
            out size: given num_insts = 10
            weight_splits ->
            torch.Size([80, 10, 1, 1])
            torch.Size([80, 8, 1, 1])
            torch.Size([10, 8, 1, 1])
            bias_splits ->
            torch.Size([80])
            torch.Size([80])
            torch.Size([10])
            """
            return weight_splits, bias_splits

        def produce_nums(num_layers, in_channels, channels):
            # 这里就设置了mask_head的weight和bias参数数量
            weight_nums, bias_nums = [], []
            for l in range(num_layers):
                if l == 0:
                    weight_nums.append((in_channels + 2) * channels)
                    bias_nums.append(channels)
                elif l == num_layers - 1:
                    weight_nums.append(channels * 1)
                    bias_nums.append(1)
                else:
                    weight_nums.append(channels * channels)
                    bias_nums.append(channels)
            return weight_nums, bias_nums

        def produce_inst_weight(weight, nums, batch_size):
            # resize weight from (N, max_obj, 169) to weight_new (n_inst, 169)
            weight_new = weight[0, :nums[0].item(), :]
            for i in range(1, batch_size):
                weight_new = torch.cat([weight_new, weight[i, :nums[i].item(), :]], dim=0)
            return weight_new

        def forward_mask_parallel(features, weights, biases, num_insts):
            """
            :param features
            :param weights: [w0, w1, ...]
            :param bias: [b0, b1, ...]
            :return:
            """
            assert features.dim() == 4
            n_layers = len(weights)
            x = features
            for i, (w, b) in enumerate(zip(weights, biases)):
                x = F.conv2d(
                    x, w, bias=b,
                    stride=1, padding=0,
                    groups=num_insts
                )
                if i < n_layers - 1:
                    x = F.relu(x)
            return x

        # def get_num_instances(target, batch_size):
        #     num_inst = 0
        #     for i in range(batch_size):
        #         num_inst += target[i].size(0)
        #     return num_inst

        def get_true_masks(target, nums, batch_size):
            """resize target from (N, max_per_n_inst, h, w) to weight_new (n_inst, h, w)"""
            masks = target[0, :nums[0].item(), :, :]
            for i in range(1, batch_size):
                masks = torch.cat([masks, target[i, :nums[i].item(), :, :]], dim=0)
            return masks


        # original codes
        mask_loss = 0.
        batch_size = seg_feat.size(0)
        weight = _tranpose_and_gather_feat(conv_weight, ind)

        h, w = seg_feat.size(-2), seg_feat.size(-1)
        x, y = ind % w, ind / w
        x_range = torch.arange(w).float().to(device=seg_feat.device)
        y_range = torch.arange(h).float().to(device=seg_feat.device)
        y_grid, x_grid = torch.meshgrid([y_range, x_range])


        # (n_inst, 169)
        weight_new = produce_inst_weight(weight, nums, batch_size)
        # produce feat(1, n_inst*10, H, W)
        feat_new = produce_mask_head_inputs(seg_feat, nums, x_grid, y_grid, x, y, batch_size)
        weight_nums, bias_nums = produce_nums(num_layers=3, in_channels=8, channels=8)
        weights, biases = parse_dynamic_params(weight_new, 8, weight_nums, bias_nums)

        num_inst = nums.sum().item()
        mask_scores = forward_mask_parallel(feat_new, weights, biases, num_inst).sigmoid()[0]
        mask_gts = get_true_masks(target, nums, batch_size)
        mask_loss_new = dice_loss(mask_scores.float(), mask_gts.float())

        # print(mask_scores.size(), mask_gts.size())
        # (1, 6, 128, 128) (6, 128, 128)
        # print(nums)
        # print(weight_new.size(), feat_new.size(), num_inst, mask_scores.size())
        # (28, 169) (1, 280, 128, 128) 28 (1, 28, 128, 128)
        # mask_loss =

#         for i in range(batch_size):  # the computation is quite slow...
#             num_obj = target[i].size(0)
#             conv1w, conv1b, conv2w, conv2b, conv3w, conv3b = \
#                 torch.split(weight[i, :num_obj], [(self.feat_channel + 2) * self.feat_channel, self.feat_channel,
#                                                   self.feat_channel ** 2, self.feat_channel,
#                                                   self.feat_channel, 1], dim=-1)

#             y_rel_coord = (y_grid[None, None] - y[i, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) / 128.
#             x_rel_coord = (x_grid[None, None] - x[i, :num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()) / 128.

#             feat = seg_feat[i][None].repeat([num_obj, 1, 1, 1])
#             feat = torch.cat([feat, x_rel_coord, y_rel_coord], dim=1).view(1, -1, h, w)

#             conv1w = conv1w.contiguous().view(-1, self.feat_channel + 2, 1, 1)
#             conv1b = conv1b.contiguous().view(-1)  # .flatten() -> .view(-1)
#             feat = F.conv2d(feat, conv1w, conv1b, groups=num_obj).relu()

#             conv2w = conv2w.contiguous().view(-1, self.feat_channel, 1, 1)
#             conv2b = conv2b.contiguous().view(-1)  #
#             feat = F.conv2d(feat, conv2w, conv2b, groups=num_obj).relu()

#             conv3w = conv3w.contiguous().view(-1, self.feat_channel, 1, 1)
#             conv3b = conv3b.contiguous().view(-1)  #
#             feat = F.conv2d(feat, conv3w, conv3b, groups=num_obj).sigmoid().squeeze()

#             true_mask = mask[i, :num_obj, None, None].float()
#             # print(target[i].size(), true_mask.size(), i)
#             # (7, 128, 128) (7, 1, 1)
#             mask_loss += dice_loss(feat * true_mask, target[i] * true_mask)
            # print(true_mask.size(), (feat*true_mask).size(), (target[i]*true_mask).size())
            # (7, 1, 1) (7, 128, 128) (7, 128, 128)
#         print(mask_loss, mask_loss_new)
        return mask_loss_new / batch_size


class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _tranpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
