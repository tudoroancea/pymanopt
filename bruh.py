import torch
from icecream import ic  # type: ignore


torch.set_default_dtype(torch.float64)

A = torch.diag(torch.tensor([1.0, 10.0]))
print(A)


def cost(x: torch.Tensor):
    return 0.5 * x @ A @ x


def grad_gt(x: torch.Tensor):
    return A @ x


def grad1(x: torch.Tensor):
    x.requires_grad_(True)
    g = torch.autograd.grad(cost(x), x)[0]
    x.requires_grad_(False)
    return g


def grad2(x: torch.Tensor):
    cost(x.requires_grad_(True)).backward()
    return x.requires_grad_(False).grad


def hvp_gt(x: torch.Tensor, v: torch.Tensor):
    return A @ v


def hvp1(x: torch.Tensor, v: torch.Tensor):
    x.requires_grad_(True)
    cost(x).backward(create_graph=True, retain_graph=True)
    dot = x.grad @ v
    x.grad = None
    dot.backward()
    x.requires_grad_(False)
    hvp, x.grad = x.grad, None
    return hvp


def hvp2(x: torch.Tensor, v: torch.Tensor):
    x.requires_grad_(True)
    cost(x).backward(create_graph=True, retain_graph=True)
    dot = x.grad @ v
    hvp = torch.autograd.grad(dot, x)[0]
    x.requires_grad_(False)
    x.grad = None
    return hvp


def hvp3(x: torch.Tensor, v: torch.Tensor):
    x.requires_grad_(True)
    g = torch.autograd.grad(cost(x), x, create_graph=True)[0]
    dot = g @ v
    x.grad = None
    dot.backward()
    x.requires_grad_(False)
    hvp, x.grad = x.grad, None
    return hvp


def hvp4(x: torch.Tensor, v: torch.Tensor):
    x.requires_grad_(True)
    g = torch.autograd.grad(cost(x), x, create_graph=True)[0]
    dot = g @ v
    x.grad = None
    hvp = torch.autograd.grad(dot, x)[0]
    x.requires_grad_(False)
    x.grad = None
    return hvp


x = torch.tensor([2.0, 1.0])
v = torch.tensor([1.0, 10.0])
ic(
    grad_gt(x),
    grad1(x),
    grad2(x),
    hvp_gt(x, v),
    hvp1(x, v),
    hvp2(x, v),
    hvp3(x, v),
    hvp4(x, v),
)
