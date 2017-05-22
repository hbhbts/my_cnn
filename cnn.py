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
        a = xin;
        za = [];
        delta_ba = [np.zeros(x.shape) for x in self.biases]
        delta_wa = [np.zeros(x.shape) for x in self.weights]

        #feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigma(z)
            za.append(z)
            aa.append(a)

        #backpropagation
        delta_ba[-1] = self.cost_prime(aa[-1], yin) * self.sigma_prime(za[-1])
        delta_wa[-1] = np.dot(delta_ba[-1], aa[-2].T)

        for i, d in enumerate(delta_ba[-2::-1]):
            delta_b = np.dot(self.weights[i+1].T, delta_ba[i+1]) * self.sigma_prime(za[i])
            delta_w = np.dot(delta_b, aa[i].T)
            delta_ba[i] = delta_b
            delta_wa[i] = delta_w


        #update bias and weight
        self.biases = [b - self.enta*nb for b, nb in zip(self.biases, delta_ba)]
        self.weights =[w - self.enta*nw for w, nw in zip(self.weights, delta_wa)]

    def cost_prime(self, a, y):
        return a-y

    def sigma(self, z):
        return 1/(1.0+np.e**(-z))

    def sigma_prime(self, z):
        return (1-self.sigma(z))*self.sigma(z)

    def test(self, xin, yin):
        a = xin
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.sigma(z)
        return int(np.argmax(a) == yin)




net0 = network([784, 50, 10], 0.1)
with gzip.open('mnist.pkl.gz', 'rb') as f:
    (train_data, validate_data, test_data) = cPickle.load(f)
    f.close()

for x, y in zip(train_data[0], train_data[1]):
    ny = np.zeros((10, 1))
    ny[y] = 1.0
    nx = np.reshape(x, (784, 1))
    net0.backpropagation(nx, ny)

print '# backpropagation complete'

correct_sum = 0
for x, y in zip(validate_data[0], validate_data[1]):
    nx = np.reshape(x, (784, 1))
    correct_sum += net0.test(nx, y)

print('result is %f' % (correct_sum/10000.0))


print '# test complete'

