import numpy as np
import matplotlib.pyplot as plt
from soft_clustering._skfcm._skfcm import SKFCM


def test_skfcm():
    np.random.seed(0)

    image = np.zeros((50, 50))
    image[:25, :] = 0.2
    image[25:, :] = 0.8
    image += np.random.normal(0, 0.05, image.shape)  # نویز

    X = image.reshape(-1, 1)
    shape = image.shape

    model = SKFCM(n_clusters=2, m=2.0, gamma=5.0, lambda_=0.8)
    model.fit(X, shape)

    labels = model.predict().reshape(shape)
    U = model.predict_proba().reshape(shape[0], shape[1], -1)

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
    plt.imshow(U[:, :, 0], cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_skfcm()
