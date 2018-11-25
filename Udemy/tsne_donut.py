import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def get_donut_date():
    N = 600
    R_inner = 10
    R_outer = 20

    R1 = np.random.randn(int(N/2)) + R_inner
    theta = 2 * np.pi * np.random.random(int(N/2))
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(int(N/2)) + R_outer
    theta = 2 * np.pi * np.random.random(int(N/2))
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*int(N/2) + [1]*(int(N/2)))

    return X,Y

def main():
    X,Y = get_donut_date()

    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    tsne = TSNE(perplexity=40)
    Z = tsne.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()
