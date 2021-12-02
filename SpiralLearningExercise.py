import numpy as np
import matplotlib.pyplot as plt


def generate_data(points_per_class=100, dim=2, num_classes=3, spiral_angle_span=4, spiral_width=0.1,
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
        plt.title('2D Spiral classification challenge')
    return X, y


def Loss_function(true_classification, scores, weights, lambda_regularization, loss_type='SVM',
                  regularization_penalty='L2'):
    N = len(true_classification)
    classification_indices = (range(N), true_classification)
    if loss_type == 'SVM':
        margin = (scores.T - scores[classification_indices]).T + 1
        L = 1 / N * (np.sum(np.maximum(0, margin))) - 1
        # in CS231n they use sum over j ~= true_classification, which is equivalent tp subtracting the margin once
        # for margin>0
        indicator = np.zeros(margin.shape)
        indicator[margin > 0] = 1
        dscores = np.copy(indicator)
        dscores[classification_indices] = 1 - np.sum(indicator, 1)
        dscores /= N
    elif loss_type == 'Softmax':
        exp_scores = np.exp((scores.T - np.max(scores, 1)).T)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        L = 1 / N * np.sum(-np.log(probs[classification_indices]))
        dscores = probs
        dscores[classification_indices] -= 1
        dscores /= N
    else:
        raise NotImplementedError

    if regularization_penalty == 'L2':
        L += 1 / 2 * lambda_regularization * np.sum(weights ** 2)
        dW_regularization = lambda_regularization * weights
    else:
        raise NotImplementedError
    return L, dscores, dW_regularization


def Linear_classifier(X, W, b):
    return np.dot(X, W) + b


def Linear_classifier_backpropagation(X, W, b, dscores):
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    return dW, db


def initialize_linear_W(dim, num_classes):
    W = 0.01 * np.random.randn(dim, num_classes)
    b = np.zeros((1, num_classes))
    return W, b


def train(iterations, X, y, lambda_regularization, dim, num_classes, classification_type='Linear'):
    if classification_type == 'Linear':
        W, b = initialize_linear_W(dim, num_classes)
    else:
        raise NotImplementedError
    losses = []
    for i in range(iterations):
        if classification_type == 'Linear':
            scores = Linear_classifier(X, W, b)
        else:
            raise NotImplementedError
        L, dscores, dW_regularization = Loss_function(y, scores, W, lambda_regularization)
        losses.append(L)
        if classification_type == 'Linear':
            dW, db = Linear_classifier_backpropagation(X, W, b, dscores)
        else:
            raise NotImplementedError
        dW += dW_regularization
        W += -grad_step * dW
        b += -grad_step * db
    plt.figure()
    plt.plot(losses, '.', label=classification_type + ' classifier with SVM loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.grid()
    predicted_class = np.argmax(scores, axis=1)
    print('Linear classifier training accuracy: %.2f' % (np.mean(predicted_class == y)))


if __name__ == "__main__":
    """
    Exercise follows stanford course CS231n: https://cs231n.github.io/neural-networks-case-study/
    """
    dim, num_classes = 2, 3
    iterations = 100
    grad_step = 0.2
    lambda_regularization = 1e-3

    size = 30
    plt.rcParams.update({'legend.fontsize': size * 0.75, 'figure.figsize': (12, 9), 'axes.labelsize': size,
                         'axes.titlesize': size,
                         'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75})

    X, y = generate_data(dim=dim, num_classes=num_classes)

    train(iterations, X, y, lambda_regularization, dim, num_classes, classification_type='Linear')

    plt.legend()
    plt.show()
