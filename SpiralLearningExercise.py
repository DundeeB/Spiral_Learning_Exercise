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


def train(iterations, X, y, lambda_regularization, dim, num_classes, num_layers=1, hidden_layer_size=None,
          test_over_fit=500):
    weights = []
    biases = []
    for i in range(num_layers):
        input_dim = dim if i == 0 else hidden_layer_size
        output_dim = hidden_layer_size if i < num_layers - 1 else num_classes
        weights.append(np.sqrt(2.0 / input_dim) * np.random.randn(input_dim, output_dim))
        biases.append(np.zeros((1, output_dim)))
    losses = []
    for iteration in range(iterations):
        hidden_layers = []
        current_layer = X
        for i in range(num_layers - 1):
            current_layer = np.maximum(0, np.dot(current_layer, weights[i]) + biases[i])
            hidden_layers.append(current_layer)
        scores = np.dot(current_layer, weights[-1]) + biases[-1]
        L, dscores, dW_regularization = Loss_function(y, scores, weights, lambda_regularization)

        dW, db = [], []
        dcurrent_layer = dscores
        for i in range(num_layers - 1):
            db.append(np.sum(dcurrent_layer, axis=0, keepdims=True))
            dW.append(np.dot(hidden_layers[-1 - i].T, dcurrent_layer) + dW_regularization[-1 - i])
            dcurrent_layer = np.dot(dcurrent_layer, weights[-1 - i].T)
            dcurrent_layer[hidden_layers[-1 - i] <= 0] = 0
        db.append(np.sum(dcurrent_layer, axis=0, keepdims=True))
        dW.append(np.dot(X.T, dcurrent_layer) + dW_regularization[0])
        dW.reverse()
        db.reverse()
        for i in range(num_layers):
            biases[i] = biases[i] - grad_step * db[i]
            weights[i] = weights[i] - grad_step * dW[i]
        losses.append(L)
        if iteration % test_over_fit == 0:
            predicted_class = np.argmax(scores, axis=1)
            accuracy = np.mean(predicted_class == y)
            print('Classifier training accuracy: %.3f' % (accuracy))
            if accuracy == 1:
                print('reached perfect accuracy')
                break
    return losses, weights, biases


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
    iterations = int(1e4)
    grad_step = 0.3
    lambda_regularization = 1e-3
    dof = 5e3  # d.o.f degrees of freedom

    size = 30
    plt.rcParams.update({'legend.fontsize': size * 0.75, 'figure.figsize': (12, 9), 'axes.labelsize': size,
                         'axes.titlesize': size,
                         'xtick.labelsize': size * 0.75, 'ytick.labelsize': size * 0.75})

    span_angle = 4 * np.pi
    for num_layers in range(1, 7):
        if num_layers == 1:
            # dof = D*K --> h can be anything
            hidden_layer_size = None
        else:
            # dof = D*h+h^2*(n-2)+h*K --> (n-2) h^2 + (D+K) h - dof = 0
            a, b, c = num_layers - 2, dim + num_classes, -dof
            if a == 0:
                x = -c / b
            else:
                x = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            hidden_layer_size = int(np.floor(x))
        print('Training ' + str(num_layers) + ' layers, hidden layer size is ' + str(hidden_layer_size))
        X, y = generate_data(dim=dim, num_classes=num_classes, spiral_angle_span=span_angle, plot=False)
        losses, weights, biases = train(iterations, X, y, lambda_regularization, dim, num_classes,
                                        num_layers=num_layers, hidden_layer_size=hidden_layer_size)


        def predict(X):
            current_layer = X
            for i in range(num_layers - 1):
                current_layer = np.maximum(0, np.dot(current_layer, weights[i]) + biases[i])
            scores = np.dot(current_layer, weights[-1]) + biases[-1]
            return np.argmax(scores, axis=1)


        plot_decision_boundaries(X, lambda X: predict(X))
        plt.title(str(num_layers) + ' layers')
        plt.figure(10)
        plt.plot(losses, '.', label=str(num_layers) + ' layers')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.show()
# Possible future excercizes:
# TODO: input data division to batches
# TODO: centering and normalizing?
# TODO: try batch normalization?
