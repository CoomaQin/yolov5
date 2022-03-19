import torch


tensor1 = torch.randn(3, 4)
tensor2 = torch.FloatTensor([0, 1, 0])
tensor2 = torch.reshape(tensor2, (-1, 1))
print(tensor2.shape)
print(tensor1.shape)
print(torch.mul(tensor2, tensor1))