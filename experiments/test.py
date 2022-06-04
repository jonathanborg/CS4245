import torch as th

x = th.FloatTensor([1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
y = th.FloatTensor([1, 1, 1, 1, 1])

loss = th.nn.BCELoss(reduction='mean')
print(loss(x, y))
