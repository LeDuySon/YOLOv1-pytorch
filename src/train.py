import os
import copy 
import math 
import glob
from pyhocon import ConfigFactory
import torch
import torch.optim as optim
from torch.utils.data import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from model.data import GlobalWheatDataset, DataLoader, collate_fn
from model.model import YOLOv1
from utils.utils import BoundingBox, YOLOv1_loss, YOLOv1_loss_old, generate_ground_truth_tensor


class Trainer:
    def __init__(self, device="cpu", lr=1e-5, epochs=10, logging_step=50, eval_step=500, batch_size=5, grid_size=7, num_bounding_boxes=2, num_labels=1, lambda_coord=5, lambda_noobj=.5, train_csv_path=None, dev_csv_path=None, eval_csv_path=None, ckpt_path="./ckpt", last_layer_hidden_size=4096):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("Training on",device)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.logging_step = logging_step
        self.eval_step = eval_step

        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        self.ckpt_path = ckpt_path
        
        self.grid_size = grid_size
        self.num_bounding_boxes = num_bounding_boxes
        self.num_labels = num_labels
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.model = YOLOv1(grid_size=grid_size, num_bounding_boxes=num_bounding_boxes, num_labels=num_labels, last_layer_hidden_size=last_layer_hidden_size)
        self.model.to(self.device)

        train_ds = GlobalWheatDataset(
                mapping_csv_path=conf["train_csv_path"],
                image_dir=conf["image_dir"]
            )
        dev_ds = GlobalWheatDataset(
                mapping_csv_path=conf["dev_csv_path"],
                image_dir=conf["image_dir"]
            )
        self.train_dataset = DataLoader(
            dataset=train_ds,
            sampler=BatchSampler(
                SequentialSampler(train_ds), batch_size=self.batch_size, drop_last=False
            ),
            collate_fn=collate_fn,
            device=self.device
        )
        self.dev_dataset = DataLoader(
            dataset=dev_ds,
            sampler=BatchSampler(
                SequentialSampler(dev_ds), batch_size=self.batch_size, drop_last=False
            ),
            collate_fn=collate_fn,
            device=self.device
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr) 
        self.tb_writer = SummaryWriter()
        self.global_step = 0

        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=self.epochs, steps_per_epoch=len(self.train_dataset))


    def train(self):
        ckpts = glob.glob(f"{self.ckpt_path}/*.pt")
        print(ckpts)
        max_ckpt_index = -1
        best_ckpt_path = None
        for ckpt in ckpts:
            ckpt_index = int(ckpt.split("-")[-1].split(".")[0])
            if ckpt_index > max_ckpt_index:
                max_ckpt_index = ckpt_index
                best_ckpt_path = ckpt

        begin_epoch = 0
        # If exist checkpoint, load from the latest checkpoint
        if best_ckpt_path:
            self.load_from_ckpt(best_ckpt_path)
            begin_epoch = max_ckpt_index+1
            self.global_step = begin_epoch * len(self.train_dataset)


        for epoch in range(begin_epoch, self.epochs):
            print(f"BEGIN EPOCH {epoch}")
            
            accumulate_loss = 0
            for idx, sample in enumerate(self.train_dataset):
                print(f"ITEM #{idx}/{len(self.train_dataset)}")
                self.optimizer.zero_grad()

                
                image_tensors, bounding_boxes = sample

                output = self.model(image_tensors)

                ground_truth = torch.zeros(len(image_tensors), self.grid_size, self.grid_size, 5*self.num_bounding_boxes+self.num_labels)

                for sample_idx in range(len(image_tensors)):
                    print("#",sample_idx,len(image_tensors))
                    # bounding_boxes[sample_idx] is now a list of bounding boxes. Each bounding box has x,y,w,h normed by image size
                    sample_truth = generate_ground_truth_tensor(bounding_boxes[sample_idx], self.grid_size, self.num_bounding_boxes, self.num_labels)
                    ground_truth[sample_idx] = sample_truth

                ground_truth = ground_truth.to(self.device)
                # ground_truth = bounding_boxes
                
                # ground_truth = [[BoundingBox(0.5,0.5,0.1,0.1,class_id=0), BoundingBox(0.9,0.9,0.1,0.1,class_id=0)],[BoundingBox(0.2,0.2,0.1,0.1,class_id=0)]]
                

                loss = YOLOv1_loss(output, ground_truth, grid_size=self.grid_size, num_labels=self.num_labels, num_bounding_boxes=self.num_bounding_boxes, lambda_coord=self.lambda_coord, lambda_noobj=self.lambda_noobj)
                # loss = YOLOv1_loss_old(output, ground_truth, grid_size=self.grid_size, num_labels=self.num_labels, num_bounding_boxes=self.num_bounding_boxes, lambda_coord=self.lambda_coord, lambda_noobj=self.lambda_noobj)
                
                accumulate_loss += loss


                
                print(f"\nLOSS: {loss}\n\n")
                if math.isnan(loss):
                    print("BREAK")
                    break
                
                loss.backward()
                self.optimizer.step()

                if (self.global_step+1) % self.logging_step == 0:
                    self.tb_writer.add_scalar("loss/train", accumulate_loss/self.logging_step, self.global_step/self.logging_step)
                    self.tb_writer.add_scalar("epoch", self.global_step/len(self.train_dataset), self.global_step)
                    self.tb_writer.add_scalar("lr", self.lr, self.global_step)
                    accumulate_loss = 0
                
                if (self.global_step+1) % self.eval_step == 0:
                    self.evaluate(step=self.global_step/self.eval_step)

                self.global_step += 1
            
            if (epoch+1) % 50 == 0:
                self.save_ckpt(epoch)

    def evaluate(self, step=0):
        print("Evaluating")
        self.model.eval()

        accumulate_loss = 0
        for idx, sample in enumerate(self.dev_dataset):
            print(f"ITEM #{idx}/{len(self.dev_dataset)}")
                
            image_tensors, bounding_boxes = sample

            output = self.model(image_tensors)

            ground_truth = torch.zeros(len(image_tensors), self.grid_size, self.grid_size, 5*self.num_bounding_boxes+self.num_labels)

            for sample_idx in range(len(image_tensors)):
                print("#",sample_idx,len(image_tensors))
                sample_truth = generate_ground_truth_tensor(bounding_boxes[sample_idx], self.grid_size, self.num_bounding_boxes, self.num_labels)
                ground_truth[sample_idx] = sample_truth

            ground_truth = ground_truth.to(self.device)

            loss = YOLOv1_loss(output, ground_truth, grid_size=self.grid_size, num_labels=self.num_labels, num_bounding_boxes=self.num_bounding_boxes, lambda_coord=self.lambda_coord, lambda_noobj=self.lambda_noobj)
            # loss = YOLOv1_loss_old(output, ground_truth, grid_size=self.grid_size, num_labels=self.num_labels, num_bounding_boxes=self.num_bounding_boxes, lambda_coord=self.lambda_coord, lambda_noobj=self.lambda_noobj)
                
            accumulate_loss += loss
        
        accumulate_loss /= len(self.dev_dataset)
        print(f"Dev Loss {accumulate_loss}")
        self.tb_writer.add_scalar("loss/dev", accumulate_loss, step)
        self.model.train()

    def load_from_ckpt(self,ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.model.load_state_dict(ckpt["model"])


    def save_ckpt(self,epoch,ckpt_path=None):
        if not ckpt_path:
            ckpt_path = os.path.join(self.ckpt_path, f"ckpt-epoch-{epoch}.pt")

        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
        }, ckpt_path)
    

if __name__ == "__main__":
    conf = ConfigFactory.parse_file("model.conf").yolov1
    print(conf)
    trainer = Trainer(
        device=conf["device"],
        lr=conf["lr"],
        epochs=conf["epochs"],
        logging_step=conf["logging_steps"],
        eval_step=conf["eval_steps"],
        grid_size=conf["grid_size"],
        batch_size=conf["batch_size"],
        num_bounding_boxes=conf["num_bounding_boxes"],
        num_labels=conf["num_labels"],
        lambda_coord=conf["lambda_coord"],
        lambda_noobj=conf["lambda_noobj"],
        train_csv_path=conf["train_csv_path"],
        dev_csv_path=conf["dev_csv_path"],
        eval_csv_path=conf["eval_csv_path"],
        ckpt_path=conf["ckpt_path"],
        last_layer_hidden_size=conf["last_layer_hidden_size"]
    )

    trainer.train()