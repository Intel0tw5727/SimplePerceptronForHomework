# coding=utf8
from SimplePerceptron import SimplePerceptron
import numpy as np

# ## Loading Data
# separatable point
import pandas as pd
p1 = pd.read_csv("point1.csv")
p2 = pd.read_csv("point2.csv")


# ## Preprocessing
# convert vector to extend vector
P1,P2 = p1.copy(),p2.copy()
P1['b'] = np.array([1] * len(P1.index))
P2['b'] = np.array([1] * len(P2.index))
P2 *= -1
point = np.r_[np.array(P1),np.array(P2)]

# ## Learning
a = SimplePerceptron()
ans = a.train(point)


# ## Visualization
import matplotlib.pyplot as plt

x1, x2 = np.array(p1.T.copy()), np.array(p2.T.copy())

x = np.arange(0,3,0.1)
y = (ans[0] * x + ans[2]) / (-ans[1])
plt.plot(x,y,"r-")
plt.plot(x1[0], x1[1],"o")
plt.plot(x2[0], x2[1], "o")
plt.show()
plt.savefig("fig.pdf")
