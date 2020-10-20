from keras.datasets import mnist
from conv import Conv
from maxpool import Maxpool
from softmax import Softmax
import numpy as np
import cv2 as cv2

(trainX, trainY), (testX, testY) = mnist.load_data()

train_images = trainX[:1000]
train_labels = trainY[:1000]
test_images = testX[:1000]
test_labels = testY[:1000]

conv = Conv(8)
maxpool = Maxpool()
softmax = Softmax(13*13*8, 10)
print('Initialised Network!')


def forward(image, label):

    out = conv.forward((image/255) - 0.5)
    out = maxpool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if label == np.argmax(out) else 0

    return out, loss, acc


def train(image, label, learn_rate=0.005):

    out, loss, acc = forward(image, label)
    gradients = np.zeros((10))
    gradients[label] = -1/out[label]
    gradients = softmax.backdrop(gradients, learn_rate)
    gradients = maxpool.backdrop(gradients)
    _ = conv.backprop(gradients, learn_rate)

    return loss, acc


def start(train_images, train_labels):
    loss = 0
    acc = 0

    print('Started training...')

    for epoch in range(3):
        print("Epoch Number : ", epoch)

        permutations = np.random.permutation(len(train_images))
        train_images = train_images[permutations]
        train_labels = train_labels[permutations]
        for i, (image, label) in enumerate(zip(train_images, train_labels)):
            curr_loss, curr_acc = train(image, label, 0.005)

            loss += curr_loss
            acc += curr_acc

            if i % 100 == 99:
                print('After %d steps, loss = %f, accuracy = %d%%' %
                      (i+1, loss/100, acc))
                loss = 0
                acc = 0

    print('Training done!')


def test(test_images, test_labels):
    print('Starting test....')

    loss = 0
    acc = 0
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        _, curr_loss, curr_acc = forward(image, label)
        loss += curr_loss
        acc += curr_acc

    num_tests = len(test_images)
    print("Results--")
    print("Loss = ", loss/num_tests)
    print("Accuracy in percent = ", (acc/num_tests)*100)


def manual_check(testX, testY):
    while(1):
        print('Do you want a manual check?')
        response = int(input('1 for yes, 0 for no: '))
        if response == 0:
            return
        idx = np.random.randint(low=0, high=testX.shape[0])
        image = testX[idx]
        label = testY[idx]
        cv2.imwrite('test.png', image)
        out, _, _ = forward(image, label)
        predicted_label = np.argmax(out)

        print('Predicted Label =', predicted_label)
        print('Actual Label =', label)


start(train_images, train_labels)
test(test_images, test_labels)
manual_check(testX, testY)
