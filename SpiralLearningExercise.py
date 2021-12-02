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
        plt.figure()
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
        dW_regularization = []
        for W in weights:
            L += 1 / 2 * lambda_regularization * np.sum(W ** 2)
            dW_regularization.append(lambda_regularization * W)
    else:
        raise NotImplementedError
    return L, dscores, dW_regularization


def train(iterations, X, y, lambda_regularization, dim, num_classes, classification_type='Linear',
          hidden_layer_size=None, test_over_fit=100):
    losses = []
    if classification_type == 'Linear':
        W = 0.01 * np.random.randn(dim, num_classes)
        b = np.zeros((1, num_classes))
        for i in range(iterations):
            scores = np.dot(X, W) + b
            L, dscores, dW_regularization = Loss_function(y, scores, [W], lambda_regularization)
            dW = np.dot(X.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)
            dW += dW_regularization[0]
            W += -grad_step * dW
            b += -grad_step * db
            losses.append(L)
        return losses, W, b
    elif classification_type == '3LayerNeuralNetwork':
        W1 = 0.01 * np.random.randn(dim, hidden_layer_size)
        b1 = np.zeros((1, hidden_layer_size))
        W2 = 0.01 * np.random.randn(hidden_layer_size, hidden_layer_size)
        b2 = np.zeros((1, hidden_layer_size))
        W3 = 0.01 * np.random.randn(hidden_layer_size, num_classes)
        b3 = np.zeros((1, num_classes))
        for i in range(iterations):
            first_layer = np.maximum(0, np.dot(X, W1) + b1)
            second_layer = np.maximum(0, np.dot(first_layer, W2) + b2)
            scores = np.dot(second_layer, W3) + b3
            L, dscores, dW_regularization = Loss_function(y, scores, [W1, W2, W3], lambda_regularization)
            dW3 = np.dot(second_layer.T, dscores) + dW_regularization[2]
            db3 = np.sum(dscores, axis=0, keepdims=True)
            dsecond_layer = np.dot(dscores, W3.T)
            dsecond_layer[second_layer <= 0] = 0
            dW2 = np.dot(first_layer.T, dsecond_layer) + dW_regularization[1]
            db2 = np.sum(dsecond_layer, axis=0, keepdims=True)
            dfirst_layer = np.dot(dsecond_layer, W2.T)
            dfirst_layer[first_layer <= 0] = 0
            dW1 = np.dot(X.T, dfirst_layer) + dW_regularization[0]
            db1 = np.sum(dfirst_layer, axis=0, keepdims=True)
            W1 += -grad_step * dW1
            b1 += -grad_step * db1
            W2 += -grad_step * dW2
            b2 += -grad_step * db2
            W3 += -grad_step * dW3
            b3 += -grad_step * db3
            losses.append(L)
            if i % test_over_fit == 0:
                predicted_class = np.argmax(scores, axis=1)
                accuracy = np.mean(predicted_class == y)
                print('Classifier training accuracy: %.3f' % (accuracy))
                if accuracy == 1:
                    print('reached perfect accuracy')
                    break
        return losses, W1, b1, W2, b2, W3, b3
    else:
        raise NotImplementedError


def plot_decision_boundaries(X, predict_func, resolution=0.01):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    plt.figure(figsize=(10, 8))

    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')


if __name__ == "__main__":
    """
    Exercise follows stanford course CS231n: https://cs231n.github.io/neural-networks-case-study/
    """
    dim, num_classes = 2, 7
    iterations = int(2e3)
    grad_step = 0.2
    lambda_regularization = 1e-3

    size = 30
    plt.rcParams.update({'legend.fontsize': size * 0.75, 'figure.figsize': (12, 9), 'axes.labelsize': size,
                         'axes.titlesize': size,
                         'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75})

    X, y = generate_data(dim=dim, num_classes=num_classes, spiral_angle_span=2 * np.pi, plot=False)
    losses_linear, W, b = train(iterations, X, y, lambda_regularization, dim, num_classes, classification_type='Linear')
    plot_decision_boundaries(X, lambda X: np.argmax(np.dot(X, W) + b, axis=1))

    X, y = generate_data(dim=dim, num_classes=num_classes, spiral_angle_span=2 * np.pi, plot=False)
    losses_neural, W1, b1, W2, b2, W3, b3 = train(iterations, X, y, lambda_regularization, dim, num_classes,
                                                  classification_type='3LayerNeuralNetwork', hidden_layer_size=100)


    def predict(X):
        first_layer = np.maximum(0, np.dot(X, W1) + b1)
        second_layer = np.maximum(0, np.dot(first_layer, W2) + b2)
        scores = np.dot(second_layer, W3) + b3
        return np.argmax(scores, axis=1)


    plot_decision_boundaries(X, lambda X: predict(X))
    plt.figure()
    for losses in [losses_linear, losses_neural]:
        plt.plot(losses, '.')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.show()
