import torch
train = torch.tensor([[1,1,1],
                     [4,5,6],
                     [1,1,1],
                     [10,11,13]])
val = torch.tensor([[[1,1,1],
                     [4,5,6],
                     [1,1,1],
                     [10,11,13]],
                     [[1,1,1],
                     [4,5,6],
                     [1,1,1],
                     [10,11,13]]]
                     )

print(train.shape)
mask = torch.all(train != 1, dim=-1)
#Use the mask to select non-empty rows
print(mask.shape)
train = train[mask]

print(train.shape)
print(train)


print("+++++")

print(val.shape)
mask = torch.all(val != 1, dim=2)
print(mask.shape)
#Use the mask to select non-empty rows

appo = val[mask]
val = appo.view(val.shape[0], -1, val.shape[2])
print(val.shape)
print(val)