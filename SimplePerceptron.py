import numpy as np

class SimplePerceptron:
    def __init__(self, W=np.round(np.random.rand(3)*10 / 2), step=1, epochs=10):
        self.W = W
        self.step = step
        self.epochs = epochs
        print("Initial Weight => {}".format(self.W))

    def predict(self, point):
        return np.where(np.dot(point,self.W.T)<=0)[0]

    def train(self, point):
        for i in range(self.epochs):
            if len(self.predict(point)) == 0:
                break
            print("epoch {}" .format(i+1))
            self.w_ = self.W
            self.w_ += point[self.predict(point)[0]] * self.step
            print("W{} => {}" .format(i+1, self.w_))
        return self.w_
