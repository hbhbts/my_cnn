import gzip
import cPickle
import numpy as np


class network:
    def __init__(self, sizes, enta):
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
        self.size = len(sizes)
        self.enta = enta

    def backpropagation(self, xin, yin):
        aa = [xin];
        za = [];
        delta_ba = [np.zeros(x.shape) for x in self.biases]
        delta_wa = [np.zeros(x.shape) for x in self.weights]

        #feedforward
        for i in range(1, self.size):
            z = np.dot(self.weights[i-1], aa[i-1].T) + self.biases[i-1]
            za.append(z)
            a = self.sigma(z)
            aa.append(a.T)

        #backpropagation
        delta_ba[-1] = self.cost_prime(aa[-1].T, yin) * self.sigma_prime(za[-1])
        delta_wa[-1] = np.dot(delta_ba[-1], aa[-2])

        for j in range(self.size-2-1, -1, -1):
            delta_b = np.dot(self.weights[j+1].T, delta_ba[j+1]) * self.sigma_prime(za[j])
            delta_w = np.dot(delta_b, aa[j])
            delta_ba[j] = delta_b
            delta_wa[j] = delta_w

        #update bias and weight
        self.biases = [b - self.enta*nb for b, nb in zip(self.biases, delta_ba)]
        self.weights =[w - self.enta*nw for w, nw in zip(self.weights, delta_wa)]

    def cost_prime(self, a, y):
        return a-y

    def sigma(self, z):
        return 1/(1.0+np.e**(-z))

    def sigma_prime(self, z):
        return (self.sigma(z)-1)*self.sigma(z)

    def test(self, xin, yin):
        a = [xin];
        for i in range(1, self.size):
            z = np.dot(self.weights[i-1], a[i-1].T) + self.biases[i-1]
            s = self.sigma(z)
            a.append(s.T)
        b = yin - a[-1].T
        print a[-1].T
        loss = 0.5 * np.dot(b.T, b)
        print loss


net0 = network([784, 20, 10], 0.1)
with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    (train_data, validate_data, test_data) = cPickle.load(f)
    f.close()

for x, y in zip(train_data[0], train_data[1]):
    ny = np.zeros((10, 1))
    ny[y] = 1.0
    nx = np.reshape(x, (1, 784))
    net0.backpropagation(nx, ny)

print '# backpropagation complete'

i = 0
for x, y in zip(validate_data[0], validate_data[1]):
    i += 1
    if i > 10:
        exit()
    ny = np.zeros((10, 1))
    ny[y] = 1.0
    nx = np.reshape(x, (1, 784))
    net0.test(nx, ny)

print '# test complete'

