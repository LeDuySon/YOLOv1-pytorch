from data import GlobalWheatData, preprocess
from model import YOLOD, testloss
from loss import calc_loss
import torch
import torch.optim as optim
from infer import predict
link = "../../yolo-pytorch/data/global-wheat-detection/train.csv"

image_link = "../../yolo-pytorch/data/global-wheat-detection/train"

PATH = "modelweight"

"""Model training hyperparameters"""
epochs = 200
batch_size = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
lr = 1e-3

wheatData = GlobalWheatData(link, image_link, preprocess)
train_data = torch.utils.data.DataLoader(wheatData, batch_size = 1, shuffle = True)

def save_checkpoint(path, model, optimizers, loss, epoch, best_acc):
  torch.save({
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizers.state_dict(),
      "loss": loss,
      "best_acc": best_acc,
      "epoch": epoch
  }, path)
  
class OutputHook:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
        
def get_hook(model):
  model.conv1.register_forward_hook(hook)

# model = YOLOD().to(device)
model = testloss()
# model.train()
optimizers = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
# isSave = False
# if(isSave):
#     checkpoint = torch.load(SAVE_PATH)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizers.load_state_dict(checkpoint["optimizer_state_dict"])
#     previous_loss = checkpoint["loss"]
#     best_acc = checkpoint["best_acc"]
# # print(f"previous_loss: {previous_loss}")
# image = ""
# for epoch in range(epochs):
#     print("Epoch: " + str(epoch))
#     for i, data in enumerate(train_data):
#         if(i != 0):
#             break
#         X, y, image = data[0].cuda(), data[1].cuda(), data[2]
#         print(image)
#         optimizers.zero_grad()
#         out = model(X)
#         loss = calc_loss(out.float(), y.float(), device = device)
#         print("Loss: %.3f" % (loss))
#         loss.backward()
#         print("test")
#         optimizers.step()


# predict(model, str(image[0]), True)

X = torch.Tensor([[0.3, -0.4, -0.3, 0.7], [0.7, 0.2, 0.5, 0.2]])
label = [torch.Tensor([[[[1, 0.5, 0.8, 0.3, 0.4, 1, 0.5, 0.8, 0.3, 0.4, 1], [0, 0, 0, 0, 0, 0,0, 0, 0,0, 0], [1, 0.5, 0.8, 0.3, 0.4, 1, 0.5, 0.8, 0.3, 0.4, 1]]]])]
for epch in range(epochs):
    for x, y in zip(X, label):
        optimizers.zero_grad()
        out = model(x)
        loss = calc_loss(out, y)
        loss.backward()
        print("Loss: %.3f" % loss)
        optimizers.step()
        print(x, "---> ", out)
        

    





