import torch.nn.functional as F 
import torch 
import torch.nn as nn
from config import noobject_scale, coord_scale, S
def calc_loss(pred_tensor, target_tensor, num_pred = 11, device = 'cpu'):
    """calculate loss func

    Args:
        pred_tensor ([tensor]): B*S*S*(B*5+C)
        target_tensor ([tesor]): B*S*S*(5+C) -> [c, x, y, w, h, prob]
    """
    
    #classification loss
    num_grid = S
    num_batch = pred_tensor.shape[0]
    coo_mask = target_tensor[..., 0] == 1 # -> B*S*S
    noo_mask = target_tensor[..., 0] == 0 # -> B*S*S
    
    noo_pred_tensor = pred_tensor[noo_mask] # -> (B*S*S)*(B*5+C)
    noo_target_tensor = target_tensor[noo_mask] # -> (B*S*S)*(B*5+C)
    # print(pred_tensor.shape)
    coo_pred_tensor = pred_tensor[coo_mask] # -> (B*S*S)*(B*5+C)
    # print(coo_pred_tensor.shape)
    # print("coo_pred: ", coo_pred_tensor)
    coo_target_tensor = target_tensor[coo_mask] # -> (B*S*S)*(B*5+C)
    
    coo_target_conf = coo_target_tensor[:, 0]
    # bounding box coord 
    box_pred = torch.cat((coo_pred_tensor[..., 1:5], coo_pred_tensor[..., 6:10]), 1)
    print(box_pred)
    
    box_target = torch.cat((coo_target_tensor[..., 1:5], coo_target_tensor[..., 6:10]), 1)
    print(box_target)
    # print(box_target.shape)
    box_target = box_target.contiguous().view(-1, 4)
    box_pred = box_pred.contiguous().view(-1, 4)
    # print(box_pred.shape)
    coo_response_mask = torch.zeros_like(box_target[:, 0].squeeze(-1)).to(device)
    # print(coo_response_mask.shape)
    iou_resbox = torch.zeros_like(box_pred[:, 0].squeeze(-1))
    for i in range(0, box_target.shape[0], 2):
        box_grid = box_pred[i:i+2] # x_center, y_center, w, h
        pbox_xymin = box_grid[:, :2]/num_grid - box_grid[:,2:]/2
        pbox_xymax = box_grid[:,:2]/num_grid + box_grid[:,2:]/2
        pbox_merge = torch.cat((pbox_xymin, pbox_xymax), 1).type(torch.FloatTensor) # xmin, ymin, xmax, ymax`
        box_gt = box_target[i:i+2] # x_center, y_center, w, h
        gbox_xymin = box_gt[:, :2]/num_grid - box_gt[:, 2:]/2
        gbox_xymax = box_gt[:,:2]/num_grid + box_gt[:, 2:]/2
        gbox_merge = torch.cat((gbox_xymin, gbox_xymax), 1).type(torch.FloatTensor)
        iou_max, index = iou(gbox_merge, pbox_merge).max(0)
        iou_resbox[i+index] = iou_max.data
        coo_response_mask[i+index] = 1
        
    
    # coord loss 
    
    tmp = coo_response_mask==1
    if(device == "cuda"):
        coo_response_mask = torch.cuda.BoolTensor(tmp)
    else:
        coo_response_mask = torch.BoolTensor(tmp)
        
    response_box = box_pred[coo_response_mask]
    print(response_box)
    response_box[response_box < 0] = 0
    gt_box = box_target[coo_response_mask]
    return response_box, iou_resbox
    iou_resbox = iou_resbox[coo_response_mask]

    coord_loss = F.mse_loss(response_box[:, :2], gt_box[:, :2]) + F.mse_loss(torch.sqrt(response_box[:,2:]), torch.sqrt(gt_box[:,2:]))
    coo_pred_conf = torch.cat((coo_pred_tensor[..., 0].unsqueeze(0).transpose(1, 0), coo_pred_tensor[..., 5].unsqueeze(0).transpose(1, 0)), 1).view(-1, 1)
    coo_pred_conf_res = coo_pred_conf[coo_response_mask].squeeze(1)
    coo_conf_loss = F.mse_loss(coo_pred_conf_res, iou_resbox)
             
    # compute classification loss
    class_pred = coo_pred_tensor[..., 10:]
    class_target = coo_target_tensor[..., 10:]
    class_loss = F.mse_loss(class_target, class_pred)                                                                                                
    
    
    
    # Compute no object confidence loss
    noo_conf_mask = torch.zeros_like(noo_pred_tensor) # -> (B*S*S)*(B*5+C)
    # mask for confidence score
    noo_conf_mask[..., 0] = 1
    noo_conf_mask[..., 5] = 1
    if(device == "cuda"):
        noo_conf_mask = torch.cuda.BoolTensor(noo_conf_mask == 1)
    else:
        noo_conf_mask = torch.BoolTensor(noo_conf_mask == 1)
    noo_conf_pred = noo_pred_tensor[noo_conf_mask]
    noo_conf_target = noo_target_tensor[noo_conf_mask]
    noobj_loss = F.mse_loss(noo_conf_pred, noo_conf_target)
    # print(noo_conf_pred, noo_conf_target)
    print(coord_loss, noobj_loss, coo_conf_loss, class_loss)
    
    return (coord_scale*coord_loss + coo_conf_loss + noobject_scale * noobj_loss + class_loss)/num_batch
    
def iou(box1, box2, threshold = 0.5):
    """Calculate iou between boxes

    Args:
        box1 ([tensor]): (N*(4)) -> xmin, ymin, xmax, ymax
        box2 ([tensor]): (N*(4)) -> xmin, ymin, xmax, ymax
        threshold (float, optional): [description]. Defaults to 0.5.
    """
    lt = torch.max(box1[:, :2], box2[:, :2]) 
    br = torch.min(box1[:, 2:], box2[:, 2:])
    
    wh = br-lt
    wh[wh<0] = 0
    # print("wh:", wh)
    inters = wh[:, 0] * wh[:, 1]
    wh_box1 = box1[:, 2:] - box1[:, :2]
    wh_box2 = box2[:, 2:] - box2[:, :2]
    union = wh_box1[:, 0] * wh_box1[:, 1] + wh_box2[:, 0] * wh_box2[:, 1] - inters 
    iou = inters / union
    return iou
    


if __name__ == '__main__':
    X = torch.zeros(2, 7, 7, 11)
    y = torch.ones(2, 7, 7, 11)
    val = calc_loss(X, y)           
    print(val)                  
    box1 = torch.Tensor([[2, 3, 5, 6], [4, 5, 7, 9]])
    box2 = torch.Tensor([[3, 4, 9, 10], [3, 4, 9, 10]])           
    print(torch.max(iou(box1, box2)))                                                  
    
    
    