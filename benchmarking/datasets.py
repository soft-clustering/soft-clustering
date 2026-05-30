from sklearn.datasets import (
    load_iris,
    load_wine,
    load_digits,
    make_blobs,
    make_moons,
    make_circles,
)


def get_dataset(name: str):

    name = name.lower()

    if name == "iris":
        return load_iris(return_X_y=True)

    if name == "wine":
        return load_wine(return_X_y=True)

    if name == "digits":
        return load_digits(return_X_y=True)

    if name == "blobs":
        return make_blobs(
            n_samples=1000,
            centers=5,
            random_state=42
        )

    if name == "moons":
        return make_moons(
            n_samples=1000,
            noise=0.05,
            random_state=42
        )

    if name == "circles":
        return make_circles(
            n_samples=1000,
            noise=0.05,
            factor=0.5,
            random_state=42
        )

    raise ValueError(
        f"Unknown dataset '{name}'"
    )