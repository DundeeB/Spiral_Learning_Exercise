import numpy as np
import matplotlib.pyplot as plt


def generate_data(points_per_class=100, dim=2, num_classes=4, spiral_angle_span=2 * np.pi, spiral_width=np.pi / 20,
                  plot=True):
    X = np.zeros((points_per_class * num_classes, dim))  # data matrix (each row = single example)
    y = np.zeros(points_per_class * num_classes, dtype='uint8')  # class labels
    for j in range(num_classes):
        ix = range(points_per_class * j, points_per_class * (j + 1))
        r = np.linspace(0.2, 1, points_per_class)  # radius
        t = 2 * np.pi * j / num_classes + np.linspace(0, spiral_angle_span, points_per_class) + np.random.randn(
            points_per_class) * spiral_width
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40)  # , cmap=plt.cm.Spectral)
        plt.show()
    return X, y


def Loss_function(true_classification, scores, weights, lambda_regularization, margin,
                  loss_type='SVM', regularization_penalty='L2'):
    N = len(true_classification)
    if loss_type == 'SVM':
        L = 1 / N * (np.sum(np.maximum(0, (scores.T - scores[range(N), true_classification]).T + margin))) - margin
        # in CS231n they use sum over j ~= true_classification, which is equivalent tp subtracting the margin once
        # for margin>0
    elif loss_type == 'Softmax':
        exp_scores = np.exp((scores.T - np.max(scores, 1)).T)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        L = 1 / N * np.sum(-np.log(probs[range(N), true_classification]))
    else:
        raise NotImplementedError

    if regularization_penalty == 'L2':
        L += 1 / 2 * lambda_regularization * np.sum(weights ** 2)
    else:
        raise NotImplementedError
    return L


if __name__ == "__main__":
    """
    Exercise follows stanford course CS231n: https://cs231n.github.io/neural-networks-case-study/
    """
    X, y = generate_data()
    print("Finished training process")
