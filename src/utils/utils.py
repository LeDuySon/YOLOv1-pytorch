import torch


class BoundingBox():
    def __init__(self,x,y,w,h,confidence=1,class_id=None):
        self.x = x # center x
        self.y = y # center y
        self.w = w # width
        self.h = h # height
        self.confidence = confidence # Confidence of prediction
        self.class_id = class_id
    
    def __str__(self):
        return f"BoundingBox({self.x},{self.y},{self.w},{self.h},{self.confidence},{self.class_id})"

    @staticmethod
    def calculate_iou(box1, box2, grid_size=1, box1_is_norm=True, box2_is_norm=True):
        """Calculate iou score of 2 box. This function will de-normalize the coordinate then calculate the score

        Args:
            box1 (BoundingBox): Bounding box with coordinate normalized
            box2 (BoundingBox): Bounding box with coordinate normalized

        Returns:
            float: intersection over union score
        """
        if box1 == None or box2 == None:
            return 0

        print("iou box1",box1)
        print("iou box2",box2)
        print()
        box1_x, box1_y, box2_x, box2_y = box1.x, box1.y, box2.x, box2.y
        if box1_is_norm:
            box1_x = box1_x * grid_size
            box1_y = box1_y * grid_size
        if box2_is_norm:
            box2_x *= box2_x * grid_size
            box2_y *= box2_y * grid_size

        intersection_x1 = max(box1_x - box1.w / 2, box2_x - box2.w / 2)
        intersection_y1 = max(box1_y - box1.h / 2, box2_y - box2.h / 2)
        intersection_x2 = min(box1_x + box1.w / 2, box2_x + box2.w / 2)
        intersection_y2 = min(box1_y + box1.h / 2, box2_y + box2.h / 2)
        intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2-intersection_y1,0)
        # print("Intersection x1",intersection_x1)
        # print("Intersection y1",intersection_y1)
        # print("Intersection x2",intersection_x2)
        # print("Intersection y2",intersection_y2)
        # print("area",intersection_area)
        union_area = box1.w * box1.h + box2.w * box2.h - intersection_area

        iou_score = 0
        if union_area != 0:
            iou_score = intersection_area / union_area
        
        print("inter_area, union_area, iou score",intersection_area, union_area, iou_score)
        return iou_score


def YOLOv1_loss(model_output, ground_truth_tensor, grid_size=7, num_labels=1, num_bounding_boxes=1, lambda_coord=5, lambda_noobj=.5):
    """Calculate loss base on output tensor and ground truth bounding boxes.

    Args:
        model_output (tensor): The model output. Shape batch_size * grid_size * grid_size * (5 * num_bounding_boxes + num_labels)
        ground_truth (tensor of BoundingBox): The list of ground truth (have to be normalized)
        grid_size (int): grid size of model
        num_bounding_boxes (int): number of bounding box per grid size
        num_labels (int): number of labels being predicted
        lambda_coord (int, optional): confidence. Defaults to 5.
        lambda_noobj (float, optional): no object rate. Defaults to .5.

    Returns:
        float: Loss of the function
    """

    loss = 0
    batch_size = 0
    
    for idx, sample in enumerate(model_output):
        sample_ground_truth_tensor = ground_truth_tensor[idx]
        for grid_y, grid_y_value in enumerate(sample):
            for grid_x, grid_pred in enumerate(grid_y_value):
                grid_true = sample_ground_truth_tensor[grid_y][grid_x]
                obj_in_grid = (grid_true[4] > 0)

                for box_idx in range(num_bounding_boxes):
                    pred_segment = grid_pred[box_idx*5:(box_idx+1)*5]
                    true_segment = grid_true[box_idx*5:(box_idx+1)*5]
                    pred_bounding_box = BoundingBox((pred_segment[0]+grid_x)/grid_size, (pred_segment[1]+grid_y)/grid_size, pred_segment[2], pred_segment[3], confidence=pred_segment[4])
                    true_bounding_box = BoundingBox((true_segment[0]+grid_x)/grid_size, (true_segment[1]+grid_y)/grid_size, true_segment[2], true_segment[3], confidence=true_segment[4])
                    iou_score = BoundingBox.calculate_iou(pred_bounding_box, true_bounding_box)
                    print(f"PRED {grid_x},{grid_y},{box_idx}",pred_bounding_box)
                    print(f"TRUE {grid_x},{grid_y},{box_idx}",true_bounding_box)
                    print(f"IOU {iou_score}")
                    
                    
                    if true_segment[4] == 0:
                        print("Loss CONFIDENCE NOOBJ", lambda_noobj*(pred_segment[4]-true_segment[4])**2)
                        loss = loss + lambda_noobj*(pred_segment[4]-true_segment[4])**2
                    else:
                        print("Loss XY", lambda_coord*((pred_segment[0]-true_segment[0])**2 + (pred_segment[1]-true_segment[1])**2))
                        print("Loss WH", lambda_coord*((pred_segment[2]**(1/2)-true_segment[2]**(1/2))**2 + (pred_segment[3]**(1/2)-true_segment[3]**(1/2))**2))
                        print("Loss CONFIDENCE OBJ", (pred_segment[4]-true_segment[4])**2)
                        
                        loss = loss + lambda_coord*((pred_segment[0]-true_segment[0])**2 + (pred_segment[1]-true_segment[1])**2)
                        loss = loss + lambda_coord*((pred_segment[2]**(1/2)-true_segment[2]**(1/2))**2 + (pred_segment[3]**(1/2)-true_segment[3]**(1/2))**2)
                        loss = loss + (pred_segment[4]-true_segment[4])**2 # confidence

                    del pred_segment
                    del true_segment
                    del pred_bounding_box
                    del true_bounding_box

                pred_class = grid_pred[-num_labels:]
                true_class = grid_true[-num_labels:]

                if obj_in_grid:
                    loss = loss + torch.sum((pred_class-true_class)**2)
                    print("Loss class", torch.sum((pred_class-true_class)**2))
                print()
                print()
        batch_size += 1

    if batch_size > 0:
        loss = loss / batch_size
    return loss


def YOLOv1_loss_old(model_output, ground_truth, grid_size, num_bounding_boxes, num_labels, lambda_coord=5, lambda_noobj=.5):
    """Calculate loss base on output tensor and ground truth bounding boxes.

    Args:
        model_output (tensor): The model output. Shape batch_size * grid_size * grid_size * (5 * num_bounding_boxes + num_labels)
        ground_truth (tensor of BoundingBox): The list of ground truth (have to be normalized)
        grid_size (int): grid size of model
        num_bounding_boxes (int): number of bounding box per grid size
        num_labels (int): number of labels being predicted
        lambda_coord (int, optional): confidence. Defaults to 5.
        lambda_noobj (float, optional): no object rate. Defaults to .5.

    Returns:
        float: Loss of the function
    """
    loss = 0
    batch_size = 0

    for idx, sample in enumerate(model_output):
        true_class_tensor = torch.zeros(grid_size, grid_size, num_labels)
        # 1 if grid contain center of a ground truth box, else 0
        obj_appear_in_cell = [[0 for i in range(grid_size)] for j in range(grid_size)]

        # Each element is the ground truth bounding box object corresponded to the grid 
        true_grid_bounding_box = [[None for i in range(grid_size)] for j in range(grid_size)] 

        true_bounding_boxes = ground_truth[idx]

        # Calculate coordinate of bounding box center grid id
        for bounding_box in true_bounding_boxes:
            grid_x = int( max(min(bounding_box.x, 1), 0) * grid_size )
            grid_x = int( max(min(grid_x, grid_size-1), 0) )
            grid_y = int( max(min(bounding_box.y, 1), 0) * grid_size)
            grid_y = int( max(min(grid_y, grid_size-1), 0) )

            if true_grid_bounding_box[grid_y][grid_x] == None:
                ground_truth_box = bounding_box
                true_class_id = bounding_box.class_id
                true_class_tensor[grid_y][grid_x][true_class_id] = 1 # assign class id tensor 
                obj_appear_in_cell[grid_y][grid_x] = 1
                
                # Ground truth x and y were normed by the entired image
                # Now we have to norm x and y by current grid position 
                ground_truth_box.x = (ground_truth_box.x - grid_x / grid_size) * grid_size
                ground_truth_box.y = (ground_truth_box.y - grid_y / grid_size) * grid_size

                true_grid_bounding_box[grid_y][grid_x] = ground_truth_box

    
        for grid_y, grid_y_value in enumerate(sample):
            for grid_x, grid_x_value in enumerate(grid_y_value):
                # grid_y_value is now a tensor containing a bounding box prediction info
                pred_bounding_boxes = grid_y_value[:5*num_bounding_boxes].reshape(num_bounding_boxes,-1)
                pred_bounding_boxes = [BoundingBox(b[0], b[1], b[2], b[3], b[4]) for b in pred_bounding_boxes]

                # prediction class for the current grid
                pred_class_tensor = grid_y_value[5*num_bounding_boxes:] 

                ground_truth_box = true_grid_bounding_box[grid_y][grid_x]

                # Now, we choose the predicted bounding box which has highest iou score
                # Set the best bounding box to the first prediction, then we will calculate the iou for every other box
                # If the grid does not contain object, then the value falls back to the first box
                best_box = pred_bounding_boxes[0]
                max_iou = BoundingBox.calculate_iou(best_box, ground_truth_box, grid_size)

                for idx in range(1,len(pred_bounding_boxes)):
                    pred_box = pred_bounding_boxes[idx]
                    current_iou = BoundingBox.calculate_iou(pred_box, ground_truth_box, grid_size=grid_size)
                    if current_iou > max_iou:
                        max_iou = current_iou
                        best_box = pred_box
                
                # Prevent square root of negative
                best_box.w = max(0, best_box.w)
                best_box.h = max(0, best_box.h)

                # Only add the loss of the bounding box if the grid contain object
                if ground_truth_box != None:
                    print("BOX LOSS 1",((best_box.x - ground_truth_box.x)**2 + (best_box.y - ground_truth_box.y)**2))
                    print("BOX LOSS 2",((best_box.w**(1/2) - ground_truth_box.h**(1/2) )**2 + (best_box.h**(1/2) - ground_truth_box.h**(1/2) )**2))
                    print("CONFIDENCE LOSS",((best_box.confidence - ground_truth_box.confidence)**2))
                    loss = loss + lambda_coord * ((best_box.x - ground_truth_box.x)**2 + (best_box.y - ground_truth_box.y)**2)
                    loss = loss + lambda_coord * ((best_box.w**(1/2) - ground_truth_box.h**(1/2) )**2 + (best_box.h**(1/2) - ground_truth_box.h**(1/2) )**2)
                    loss = loss + lambda_coord * ((best_box.confidence - ground_truth_box.confidence)**2)
                else:
                    print("NOOBJ LOSS",((not obj_appear_in_cell[grid_y][grid_x]) * torch.sum((true_class_tensor - pred_class_tensor) ** 2)))
                    loss = loss + lambda_noobj * ((best_box.confidence - 0)**2)
                loss = loss + ((not obj_appear_in_cell[grid_y][grid_x]) * torch.sum((true_class_tensor - pred_class_tensor) ** 2))
 
        batch_size += 1

    if batch_size > 0:
        loss = loss / batch_size
    return loss

# def get_responsible_tensor(pred, true_bounding_boxes, grid_size, num_bounding_boxes, num_labels):
#     """Get Iobj needed to calculate loss

#     Args:
#         pred (tensor): The model output. Shape 1 * grid_size * grid_size * (5 * num_bounding_boxes + num_labels)
#         true_bounding_boxes (tensor): The list of ground truth (have to be normalized)
#         grid_size (int): grid size of model
#         num_bounding_boxes (int): number of bounding box per grid size
#         num_labels (int): number of labels being predicted
    
#     Returns:
#         ftensor: the 1/0 tensor, shape 1xgrid_size*grid_size*(5*bounding_box+num_labels)
#     """
#     I = torch.zeros(1, grid_size, grid_size, num_labels)

    
def generate_ground_truth_tensor(true_bounding_boxes, grid_size, num_bounding_boxes, num_labels):
    """Generate tensor from ground truth bounding box (has size equal to model output)

    Args:
        true_bounding_boxes (List[BoundingBox]): List of ground truth bounding box
        grid_size (int): size of grid
        num_bounding_boxes ([type]): bounding box per grid
        num_labels ([type]): number of labels
    """
    result = torch.zeros(grid_size, grid_size, 5*num_bounding_boxes+num_labels)

    for bounding_box in true_bounding_boxes:
        grid_x = int( max(min(bounding_box.x, 1), 0) * grid_size )
        grid_x = int( max(min(grid_x, grid_size-1), 0) )
        grid_y = int( max(min(bounding_box.y, 1), 0) * grid_size)
        grid_y = int( max(min(grid_y, grid_size-1), 0) )

        grid_tensor = []
        for _ in range(num_bounding_boxes):
            grid_tensor += [bounding_box.x*grid_size-grid_x, bounding_box.y*grid_size-grid_y, bounding_box.w, bounding_box.h, bounding_box.confidence]
        class_ids = [0 for i in range(num_labels)]
        class_ids[bounding_box.class_id] = 1
        grid_tensor += class_ids

        result[grid_y][grid_x] = torch.tensor(grid_tensor)

    return result


def normalize_coordinate(img_width, img_height, x, y):
    return x / img_width, y/img_height
