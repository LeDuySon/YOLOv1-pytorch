https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
- Using AMP to reduce size of model

- Using dataloader num_worker and enable pinmemory to speed up training

- Using cuDNN autotuner to speed up training (torch.backend.cudnn.benchmark = True). 
How it works?
 -> It enables benchmark mode in cudnn.
 -> cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.  
When to use?
 -> benchmark mode is good whenever your input sizes for your network do not vary
