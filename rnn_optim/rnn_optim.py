import torch


class AccountNn(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_wallet, in_trade, price_delta, buy):
        in_trade = in_trade * price_delta
        buy = buy * (in_wallet + in_trade)
        # max_trade = in_wallet / (1 + (0.001 * 2)) needed commission
        buy = torch.minimum(in_wallet, buy)  # if buy is positive: wallet -> trade
        buy = torch.maximum(-in_trade, buy)  # if buy is negative: trade -> wallet
        in_wallet = in_wallet - buy
        in_trade = in_trade + buy
        # in_wallet -= to_trade * (0.001 * 2) commission
        return in_wallet, in_trade


class TraderNn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # device?
        self.lin1 = torch.nn.Linear(in_features=2 + 5, out_features=10)
        self.lin2 = torch.nn.Linear(in_features=10, out_features=1 + 5)

    def forward(self, in_wallet, in_trade, price, price_delta, state):
        x = torch.concat(((in_wallet / (in_wallet + in_trade)), price_delta))
        x = torch.concat((x, state))
        x = self.lin1(x).relu()
        x = self.lin2(x).tanh()
        y = x[0]
        state = x[1:]
        return y, state


def test_acc():
    from torch import as_tensor as t
    acc = AccountNn()
    wa, tr = acc(t(50.0), t(50.0), t(1.1), t(1.0))
    wa, tr = acc(t(50.0), t(50.0), t(1.1), t(0.25))
    wa, tr = acc(t(50.0), t(50.0), t(1.1), t(-0.25))
    wa, tr = acc(t(50.0), t(0.0), t(1.1), t(0.25))
    wa, tr = acc(t(0.0), t(50.0), t(1.1), t(0.25))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Available device: {device.type}")

    acc = AccountNn()
    trader = TraderNn()

    # https://pytorch.org/docs/1.9.1/generated/torch.nn.RNN.html
    # trader = torch.nn.RNN(input_size=3, hidden_size=20, num_layers=2)

    optimizer = torch.optim.SGD(trader.parameters(), lr=1e-2)

    prices = [100, 102, 104, 98, 95, 90, 99, 103, 105]
    for i_epoch in range(10):
        in_wallet = torch.as_tensor([100.0])
        in_trade = torch.as_tensor([0.0])
        state = torch.zeros([5])
        price_prev = prices[0]
        cum_money = torch.tensor([0.0])

        # print(f"--- Epoch: {i_epoch}")
        for price in prices:
            price_delta = torch.tensor([price / price_prev])
            price_prev = price
            price = torch.tensor([price])

            buy, state = trader(in_wallet, in_trade, price, price_delta, state)
            # print(f"buy: {buy.item()} in_wallet: {in_wallet.item()} in_trade: {in_trade.item()}")

            in_wallet, in_trade = acc(in_wallet, in_trade, price_delta, buy)

            cum_money = cum_money + in_wallet + in_trade

        optimizer.zero_grad()
        loss = -cum_money / len(prices)
        loss.backward()
        optimizer.step()

        # in_wallet = in_wallet.detach()
        # in_trade = in_trade.detach()
        # state = state.detach()

        print(f"Epoch: {i_epoch} loss: {(-loss).item()}")
        # print(f"trader param: {trader.lin1.weight[0, 0]}")
        # print(f"trader param grad: {trader.lin1.weight.grad[0, 0]}")

        # TODO only a single addition multiple times in a loop backward

        del in_wallet, in_trade, state, buy, cum_money, loss, price, price_delta, price_prev


if __name__ == '__main__':
    main()
