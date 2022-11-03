from __future__ import annotations
from abc import abstractmethod  # Python 3.7+
from typing import Any, List, Optional, Type

import numpy as np
import numpy.typing as npt

NpArray = npt.NDArray[np.float16 | np.float32 | np.float64]

"""
autograd: the idea

    a = Tensor(...)
    b = Tensor(...)
    c = a.dot(b)
    c.backward()

in order to compute dc/da = b and dc/db = a, `c.backward()` needs to know
how many inputs does `c` have, in this case, `a` and `b`. secondly, it
needs to know what operation combine these two inputs into `c`, in this case,
it's dot.

you see, in the end

    1. `c.grad = np.ones()`
    2. `a.grad = b`
    2. `b.grad = a`

so `Tensor.backward(leaf=True)` fills `self.grad` to `np.ones()` if this is
a leaf node. then, self must assign its inputs' grads. where does these grads
come from? of course `a.dot(b)` somehow embeds the op info into tensor `c`,
and `backward` can rely on op info to compute grads with respect to inputs `a`
and `b`. but how exactly?

it's possible to attach a `Context` object to `c`, which contains op info and inputs.
"""


class Context:
    op: Optional[Type[Op]]
    inputs: List[Tensor]

    def __init__(self, op: Optional[Type[Op]], inputs: List[Tensor]):
        self.op = op
        self.inputs = inputs

    def backward(self, grad_output: NpArray) -> NpArray | List[NpArray]:
        assert self.op is not None
        inputs_np = list(map(lambda t: t.data, self.inputs))
        return self.op.backward(inputs_np, grad_output)


class Tensor:
    data: NpArray
    grad: Optional[NpArray]
    _ctx: Optional[Context]

    def __init__(self, data: NpArray):
        self.data = data
        self.grad = None
        self._ctx = None

    def backward(self, leaf: bool = True) -> None:

        # bail out when node is root
        if self._ctx is None:
            return

        # fill self.grad with ones if it's a leaf node
        if leaf:
            self.grad = np.ones_like(self.data)

        assert self.grad is not None

        # compute grads with respect to inputs
        grads = self._ctx.backward(self.grad)
        assert grads is not None

        # propagate grads to `inputs.grad`
        if not isinstance(grads, list):
            grads = [grads]

        for t, grad in zip(self._ctx.inputs, grads):
            if t.data.shape != grad.shape:
                raise ValueError(
                    "gradient dimension {} doesn't match with tensor {}".format(
                        grad.shape, t.data.shape
                    )
                )
            t.grad = grad
            t.backward(leaf=False)

    def dot(self, rhs: Tensor) -> Tensor:
        t = Tensor(self.data.dot(rhs.data))
        t._ctx = Context(Dot, [self, rhs])
        return t

    def add(self, rhs: Tensor) -> Tensor:
        t = Tensor(self.data + rhs.data)
        t._ctx = Context(Add, [self, rhs])
        return t


class Op:
    @staticmethod
    @abstractmethod
    def forward(x: NpArray, y: NpArray) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(
        inputs: List[NpArray], grad_output: NpArray
    ) -> NpArray | List[NpArray]:
        raise NotImplementedError


class Dot(Op):
    @staticmethod
    def forward(x: NpArray, y: NpArray) -> Any:
        return x.dot(y)

    # W.dot(x) -> (W1, x2)
    # W2 = x1
    # dDot/dW = x
    # df/dW = df/dDot * dDot/dW = grad_output * x -> (W1, W2) = (W1, x1)
    #                             (W1, x2)     (x1, x2)
    # df/dx = df/dDot * dDot/dx = W * grad_output -> (x1, x2) = (W2, x2)
    #                         (W1, W2)    (W1, x2)
    @staticmethod
    def backward(
        inputs: List[NpArray], grad_output: NpArray
    ) -> NpArray | List[NpArray]:
        return [grad_output.dot(inputs[1].T), grad_output.T.dot(inputs[0])]

# f = f(x + y)
# (x + y).shape = (x1, x2) (assume no broadcast)
# df/dx = df/d(x+y) * d(x+y)/dx -> 
#         (x1,x2)     (x1 x2)
# gradient should match the size of x, scalar is simple.
# for vectorized case, assume mul 
class Add(Op):
    @staticmethod
    def forward(x: NpArray, y: NpArray) -> Any:
        return x + y

    @staticmethod
    def backward(
        inputs: List[NpArray], grad_output: NpArray
    ) -> NpArray | List[NpArray]:
        return list(map(lambda x: np.multiply(np.ones_like(x), grad_output), inputs))


if __name__ == "__main__":
    x = Tensor(np.array([2]))
    y = Tensor(np.array([3]))
    z = Tensor(np.array([8]))

    print('x =', x.data)
    print('y =', y.data)
    print('z =', z.data)

    q = x.add(y)
    f = q.dot(z)

    print("f = q * (x + y) =", f.data)

    f.backward()
    print("f.grad =", f.grad)
    print("q.grad =", q.grad)
    print("z.grad =", z.grad)
    print("x.grad =", x.grad)
    print("y.grad =", y.grad)
