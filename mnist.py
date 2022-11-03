# type: ignore

import numpy as np
from tqdm import trange
import torch
import math


def fetch(url):
    import requests, gzip, os, hashlib

    cache_dir = os.path.join(os.getcwd(), ".cache")
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    fp = os.path.join(cache_dir, hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8)


def fetch_mnist():
    X_train = (
        fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[16:]
        .reshape((-1, 28, 28))
        .astype(np.float32)
        / 255
    )
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = (
        fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[16:]
        .reshape((-1, 28, 28))
        .astype(np.float32)
        / 255
    )
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    return (X_train, Y_train), (X_test, Y_test)


class SmolNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 128, bias=False)
        self.l2 = torch.nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.log_softmax(self.l2(x), dim=1)
        return x


def torch_eval(X_test, Y_test, model):
    import torch
    from torch import tensor

    X = tensor(X_test).reshape((-1, 28 * 28))
    Y = tensor(Y_test)
    out = model(X)
    pred = out.argmax(axis=1)
    acc = (Y == pred).float().mean()
    return acc


def torch_train(X_train, Y_train):
    import torch
    from torch import tensor

    model = SmolNet()

    BATCH = 32
    # optimizer = optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accuracies = []
    losses = []
    for i in (t := trange(1000)):
        # prepare data
        samp = np.random.randint(0, X_train.shape[0], size=(BATCH))
        X = tensor(X_train[samp], requires_grad=True).reshape((-1, 28 * 28))
        Y = tensor(Y_train[samp]).long()

        # train
        optimizer.zero_grad()
        out = model(X)
        pred = out.argmax(axis=1)
        acc = (pred == Y).float().mean()
        loss = torch.nn.functional.nll_loss(out, Y)
        loss.backward()
        optimizer.step()

        # stat
        acc, loss = acc.item(), loss.item()
        accuracies.append(acc)
        losses.append(loss)
        t.set_description("loss={:.2f}, acc={:2f}".format(loss, acc))

    return model, (losses, accuracies)


def log_softmax(x):
    x_max = x.max(axis=1)
    logexpsum = x_max + np.log(np.exp(x - x_max.reshape((-1, 1))).sum(axis=1))
    return x - logexpsum.reshape((-1, 1))


def smol_forward(x, l1, l2):
    x = x.dot(l1)
    x = np.maximum(0, x)  # relu
    x = x.dot(l2)
    x = log_softmax(x)
    return x


def smol_eval(X_test, Y_test, weights):
    preds = smol_forward(X_test.reshape((-1, 28 * 28)), *weights).argmax(axis=1)
    acc = (preds == Y_test).mean()
    return acc


def kaiming_init(shape):
    dim = len(shape)
    if dim < 2:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = shape[1]
    receptive_field_size = 1
    if dim > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    std = math.sqrt(2.0) / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    w = np.random.uniform(-bound, bound, shape)
    return w


def smol_train(X_train, Y_train):
    # init weights

    # kaiming init
    l1 = kaiming_init((28 * 28, 128))
    l2 = kaiming_init((128, 10))

    # borrow initialization from pytorch
    # torch_net = SmolNet()
    # l1 = np.zeros(((28 * 28, 128)))
    # l2 = np.zeros(((128, 10)))
    # l1[:] = torch_net.l1.weight.detach().numpy().transpose()
    # l2[:] = torch_net.l2.weight.detach().numpy().transpose()

    # l1[:] = torch_l1
    # l2[:] = torch_l2
    # l1 = np.random.rand(28 * 28, 128).astype(np.float32)
    # l2 = np.random.rand(128, 10).astype(np.float32)

    BATCH = 32
    lr = 0.1

    losses = []
    accuracies = []

    for i in (t := trange(1000)):
        # forward
        # prepaer data
        samp = np.random.randint(0, X_train.shape[0], size=(BATCH))
        X = X_train[samp]
        Y = Y_train[samp]

        x = X.reshape((-1, 28 * 28))
        x_l1 = x.dot(l1)
        x_relu = np.maximum(0, x_l1)
        x_l2 = x_relu.dot(l2)  # W = trelu, x = l2, feels inverted...
        # print(l2)
        x_lsm = log_softmax(x_l2)
        loss = -x_lsm[range(x_lsm.shape[0]), Y]  # nll, take label component and neagte
        # print('loss =', loss.mean())

        # compute the accuracy on batch, to plot an accurary graph
        pred = x_lsm.argmax(axis=1)
        acc = (pred == Y).astype(np.float32).mean()

        # backward

        # gradient of NLL, i.e. dL/dout
        dlsm = np.zeros(x_lsm.shape)
        dlsm[range(dlsm.shape[0]), Y] = -1.0 / BATCH
        # print(dlsm)

        # gradient of log softmax, i.e. dL/dxl2
        # print(dlsm.sum(axis=1))
        dx_l2 = dlsm - np.exp(x_lsm) * dlsm.sum(axis=1).reshape((-1, 1))

        # gradient of dot(l2), dL/drelu and dL/dl2
        # input dtl2 (1,10), trelu (1, 128)
        # output (128, 10)
        dl2 = x_relu.T.dot(dx_l2)
        # print(x_relu)

        # input dtl2 (1, 10), l2 (128,10)
        # output shape (1, 128)
        dx_relu = dx_l2.dot(l2.T)

        # gradient of relu, dL/dtl1 = dL/drelu * drelu/dtl1
        dx_l1 = dx_relu.copy()
        dx_l1[(x_l1 < 0.0)] = 0.0

        # gradient of dot(l1), dL/dx or dL/dl1, but we only want dL/dl1, because we already reached the starting point
        # dL/dl1 = dL/dtl1 * dtl1/dl1 = dL/dtl1 * x
        # input x (1,784), dtl1 (1, 128)
        # output (784, 128)
        dl1 = x.T.dot(dx_l1)

        # SGD
        l2 = l2 - lr * dl2
        l1 = l1 - lr * dl1

        # stat
        accuracies.append(acc)
        losses.append(loss.mean())
        t.set_description("loss={:2f}, acc={:2f}".format(loss.mean(), acc))

    return (l1, l2), (losses, accuracies)


def train_everything():
    train, test = fetch_mnist()
    #
    # weights, _ = smol_train(*train)
    # print('smol eval acc', smol_eval(*test, weights))

    print(train[0][0].tolist())

    # l1, l2 = weights
    # print(l1.shape, l2.shape)
    # l1.tofile('l1.raw');
    # l2.tofile('l2.raw');

    # model, _ = torch_train(*train)
    # print("torch eval acc", torch_eval(*test, model))


def main():
    train_everything()
    # print(kaiming_init((2,2)))


if __name__ == "__main__":
    main()
