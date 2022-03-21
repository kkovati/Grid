import torch


def main():
    q = torch.tensor([0.1], requires_grad=True)

    opt = torch.optim.SGD([q], lr=1e-2)

    for e in range(10):
        x = torch.tensor([0.0])
        y = torch.tensor([0.0])
        for i in range(10):
            x = x + q
            y = y + x

        y = y / 10
        print(y.item())

        opt.zero_grad()
        loss = -y
        loss.backward()
        opt.step()

        del x, y


if __name__ == '__main__':
    main()
