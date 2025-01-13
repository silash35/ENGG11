import torch


def dydx(x, y):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def mean_square(x):
    return torch.mean(x**2)


def train(
    model, loss_fn, optimizer, epochs, t: torch.Tensor, target_loss: None | float = None
):
    loss_value = 0
    for _ in range(epochs):
        # Coloca o modelo no modo de treinamento
        model.train()

        # Calcula o loss usando a nossa função loss.
        loss = loss_fn(model, t)

        # Ajusta os valores do modelo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().numpy())
        if target_loss is not None and loss_value < target_loss:
            break

    return loss_value
