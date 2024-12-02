import matplotlib.pyplot as plt

default_dpi = 96
save_folder = "../media/"
plt.rcParams.update({"font.size": 15})


def save_or_show(filename: str | None = None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(save_folder + filename)
        plt.clf()
        plt.close()


def plot_tanks(t, tanks, labels=None, filename: str | None = None, scatter=0):
    plt.figure(figsize=(10, 4), layout="constrained", dpi=default_dpi)

    for i, tank in enumerate(tanks):
        label = labels[i] if (type(labels) is list) else labels
        if i < scatter:
            plt.scatter(t, tank, label=label, s=8, alpha=0.5)
        else:
            plt.plot(t, tank, label=label)
    plt.xlabel("Tempo / s")
    plt.ylabel("Nível / cm")
    plt.legend()

    save_or_show(filename)


def plot_loss(loss_values, epoch_count, filename: str | None = None):
    plt.figure(figsize=(10, 4), layout="constrained", dpi=default_dpi)
    plt.plot(epoch_count, loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    save_or_show(filename)
