import torch
import torch.nn as nn
import numpy as np


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.sigmoid(logits)
        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        loss = 5.0 * pos_loss + 1.0 * neg_loss

        if self.reduction == 'mean':
            #batch_size = logits.size(0)
            loss = torch.sum(loss) #/ batch_size
        return loss

class YOLO2Loss(nn.Module):
    def __init__(self, num_anchors=5, num_classes=20, input_size=416, ignore_thresh=0.5, lambda_noobj=1.0, lambda_coord=5.0):
        super(YOLO2Loss, self).__init__()
        self.anchor_size = torch.tensor([[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]])
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.input_size = input_size
        self.ignore_thresh = ignore_thresh
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.device = 'cuda'

        self.conf_loss_function = MSEWithLogitsLoss(reduction='mean')
        self.cls_loss_function = nn.CrossEntropyLoss(reduction='none')
        self.txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
        self.twth_loss_function = nn.MSELoss(reduction='none')
        self.iou_loss_function = nn.SmoothL1Loss(reduction='none')

        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
    # def conver_box(self, box, index):
    #     step = 1.0 / self.input_size
    #     box[:, 0] = index[:, 0] * step + box[:, 0] * step
    #     box[:, 1] = index[:, 1] * step + box[:, 1] * step
    #     box[:, 2] = box[:, 2] * step
    #     box[:, 3] = box[:, 3] * step
    #     return box
    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)], indexing='ij')
        #print(grid_x, grid_y)
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)


        return grid_xy, anchor_wh
    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
    def decode_xywh(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                xywh_pred : [B, H*W*anchor_n, 4] \n
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, -1, 4) * self.stride

        return xywh_pred
    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred

    def iou_score(self, bboxes_a, bboxes_b):
        """
            bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
            bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
        """
        tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
        return area_i / (area_a + area_b - area_i + 1e-14)
    # def compute_iou(self, box1, box2, index):
    #     box1 = torch.clone(box1)
    #     box2 = torch.clone(box2)
    #     box1 = self.conver_box(box1, index)
    #     box2 = self.conver_box(box2, index)
    #     x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    #     x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #     inter_w = (w1 + w2) - (torch.max(x1 + w1, x2 + w2) - torch.min(x1, x2))
    #     inter_h = (h1 + h2) - (torch.max(y1 + h1, y2 + h2) - torch.min(y1, y2))
    #     inter_h = torch.clamp(inter_h, 0)
    #     inter_w = torch.clamp(inter_w, 0)

    #     inter = inter_w * inter_h
    #     union = w1 * h1 + w2 * h2 - inter
    #     return inter / union



    def forward(self, pred, target):
        #print(target.shape) torch.Size([16, 1, 845, 11])
        #print(pred.shape) torch.Size([16, 125, 13, 13])
        B, abC, H, W = pred.size()
        target = target.squeeze(1)
        #print(target.shape) torch.Size([16, 845, 11])
        #pred = pred.view(B, H, W, self.num_anchors, -1)

        pred = pred.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)


        # print(pred.shape) torch.Size([16, 1625, 13])
        # conf_pred: confidence prediction
        #conf_pred = pred[..., :1].contiguous().view(B, H * W * self.num_anchors, 1)
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        
        # cls_pred: class prediction
        # cls_pred = pred[..., 1:1+self.num_classes].contiguous().view(B, H * W * self.num_anchors, self.num_classes)
        cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)

        # reg_pred: regression prediction (box coordinates)
        #reg_pred = pred[..., 1+self.num_classes:].contiguous().view(B, H * W * self.num_anchors, 4)
        reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        reg_pred = reg_pred.view(B, H*W, self.num_anchors, 4)
        #print(reg_pred.shape) #torch.Size([16, 169, 5, 4])

        # decode bbox
        x1y1x2y2_pred = (self.decode_boxes(reg_pred) / self.input_size).reshape(-1, 4)
        # print(x1y1x2y2_pred.shape) #torch.Size([16*845, 4])
        x1y1x2y2_gt = (target[:, :, 7:11]).reshape(-1, 4)
        # print(x1y1x2y2_gt.shape) #torch.Size([16*845, 4])

        reg_pred = reg_pred.reshape(B, H*W*self.num_anchors, 4)
        # x1y1x2y2_pred = (reg_pred.reshape(B, H * W * self.num_anchors, 4) / self.input_size).reshape(-1, 4)
        # x1y1x2y2_gt = target[:, :, 7:].reshape(-1, 4)
        # reg_pred = reg_pred.reshape(B, H * W * self.num_anchors, 4)

        # set conf target
        iou_pred = self.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).reshape(B, -1, 1)
        gt_conf = iou_pred.clone().detach()

        # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
        target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

        ''' pred_conf=conf_pred,
            pred_cls=cls_pred,
            pred_txtytwth=reg_pred,
            pred_iou=iou_pred,
            label=target'''
        
        pred_conf = conf_pred[:, :, 0]
        pred_cls = cls_pred.permute(0, 2, 1)
        pred_txty = reg_pred[:, :, :2]
        pred_twth = reg_pred[:, :, 2:]
        pred_iou = iou_pred[:, :, 0]

        gt_conf = target[:, :, 0].float()
        gt_obj = target[:, :, 1].float()
        gt_cls = target[:, :, 2].long()
        gt_txty = target[:, :, 3:5].float()
        gt_twth = target[:, :, 5:7].float()
        gt_box_scale_weight = target[:, :, 7].float()
        gt_iou = (gt_box_scale_weight > 0.).float()
        gt_mask = (gt_box_scale_weight > 0.).float()

        batch_size = pred_conf.size(0)
        # objectness loss
        obj_loss = self.conf_loss_function(pred_conf, gt_conf, gt_obj)
        #noobj_loss = self.conf_loss_function(pred_conf, torch.zeros_like(gt_conf), (gt_obj == 0).float())

        # class loss
        class_loss = torch.sum(self.cls_loss_function(pred_cls, gt_cls) * gt_mask)
        
        # box loss
        txty_loss = torch.sum(torch.sum(self.txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_mask)
        twth_loss = torch.sum(torch.sum(self.twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_mask)
        bbox_loss = txty_loss + twth_loss

        # iou loss
        iou_loss = torch.sum(self.iou_loss_function(pred_iou, gt_iou) * gt_mask)

        total_loss = obj_loss + bbox_loss + class_loss + iou_loss

        return total_loss / batch_size

# Example usage:
# pred = model_output
# target = ground_truth_tensor
# yolo_loss = YOLOLoss()
# loss = yolo_loss(pred, target)
