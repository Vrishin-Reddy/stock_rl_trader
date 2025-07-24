import matplotlib.pyplot as plt

def plot_trades(trade_log):
    # trade_log: list of (step, action, price, reward)
    steps = [t[0] for t in trade_log]
    prices = [t[2] for t in trade_log]

    plt.plot(steps, prices, label='Price')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.title('Trade Actions Over Time')
    plt.legend()
    plt.show()
