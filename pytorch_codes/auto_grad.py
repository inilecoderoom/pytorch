import torch as tr 


x = tr.rand(3, requires_grad= True)
print(x)

# y = x+2

# print(y)

# z = y*y*2
# z = z.mean()
# print(z)

# v = tr.tensor([0.1,1.0,0.001], dtype=tr.float32)

# z.backward(v)
# print(x.grad)

# x.requires_grad_(False)
# print(x)
# y = x.detach()
# print(y)

# with tr.no_grad():
#     y = x+2
#     print(y)




wt = tr.ones(4, requires_grad=True)
for each in range(3):
    model_output = (wt*3).sum()
    model_output.backward()
    print(wt.grad)


optimizer = tr.optim.SGD(wt, lt = 0.01)
optimizer.step()
optimizer.zero_grad()

# https://www.youtube.com/watch?v=c36lUUr864M
# 50.05
