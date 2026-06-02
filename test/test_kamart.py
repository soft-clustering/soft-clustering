from os import path
import sys

if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import KMART

    documents = [
        "the cat sat on the mat",
        "a dog ate my homework",
        "the cat and the dog are friends",
        "my homework is about machine learning",
        "machine learning models need data",
        "cat and dog are common pets"
    ]

    model = KMART(vigilance_param=0.75, learning_rate=0.5)
    memberships = model.fit_predict(documents)

    print("Number of documents:", memberships.shape[0])
    print("Number of clusters:", memberships.shape[1])
    print("Membership matrix:\n", memberships.toarray())
