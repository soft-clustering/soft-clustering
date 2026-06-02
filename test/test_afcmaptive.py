from os import path
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import AFCMAdaptive

    np.random.seed(0)

    image = np.zeros((64, 64))
    image[:32, :] = 0.3
    image[32:, :] = 0.7
    image += np.random.normal(0, 0.05, image.shape)  # noise

    model = AFCMAdaptive(n_clusters=2, m=2.0, k1=0.1, k2=0.1, max_iter=50)
    model.fit(image)

    labels = model.predict()
    membership = model.get_membership()

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Labels")
    plt.imshow(labels, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Membership (Class 1)")
    plt.imshow(membership[:, :, 0], cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
