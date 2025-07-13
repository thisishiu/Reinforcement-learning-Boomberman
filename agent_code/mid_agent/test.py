# Using this class after implement basic grad

class Variable:
    def __init__(self, value):
        self.value = value
        self.grad = 0.0
        self._baclward = lambda: None
        self._prev = []

    def __add__(self, other):
        if not isinstance(other, Variable): other = Variable(other)
        out = Variable(self.value + other.value)
        out._prev = [self, other]

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._baclward = _backward
        return out
    
    def __sub__(self, other):
        if not isinstance(other, Variable): other = Variable(other)
        out = Variable(self.value - other.value)
        out._prev = [self, other]

        def _backward():
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad
        out._baclward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Variable): other = Variable(other)
        out = Variable(self.value * other.value)
        out._prev = [self, other]

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._baclward = _backward
        return out
    
    def backward(self):
        ...

    def __str__(self):
        return str(self.value)

if __name__ == "__main__":
    x = Variable(2)
    y = x * 2
    y._baclward()
    print(x.grad)


