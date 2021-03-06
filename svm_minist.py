from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np


def show_some_digits(images, targets, sample_size=9, title_text='Digit {}' ):
    '''
    Visualize random digits in a grid plot
    images - array of flatten gidigs [:,784]
    targets - final labels
    '''
    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))
    img = plt.figure(1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
        plt.axis('off')
        # each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))
    plt.show()


if __name__ == "__main__":
    mnist = fetch_mldata('MNIST original', data_home='./')
    mnist.keys()
    images = mnist.data
    targets = mnist.target
    show_some_digits(images, targets)
