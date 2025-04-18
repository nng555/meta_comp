import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture

def gen_gmm_data(
    n_samples: int,
    n_components: int,
    n_features: int = 2,
    seed: int = None
):
    if seed is not None:
        np.random.seed(seed)

    # Generate random mixing proportions
    weights = np.random.dirichlet(np.ones(n_components))

    # Generate random means for each component
    means = np.random.uniform(-10, 10, (n_components, n_features))

    # Generate random covariance matrices
    covs = []
    for _ in range(n_components):
        # Create a random positive definite matrix
        A = np.random.randn(n_features, n_features)
        cov = np.dot(A, A.T) + np.eye(n_features)  # Add identity to ensure positive definiteness
        covs.append(cov)

    # Generate component assignments based on weights
    y = np.random.choice(n_components, size=n_samples, p=weights)

    # Generate samples for each component
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        component = y[i]
        X[i] = np.random.multivariate_normal(means[component], covs[component])

    return X, y

if __name__ == "__main__":

    # load or generate data
    ncomponents = 128
    logn_components = int(np.log2(ncomponents))

    data_path = 'base_data.npy'
    if not os.path.exists(data_path):
        print("Generating data")

        x, y = gen_gmm_data(
            n_samples=3000,
            n_components=ncomponents,
            n_features=64,
            seed=0,
        )
        np.save(data_path, x)

    else:
        print("Loading data")
        x = np.load(data_path)

    train_x = x[:2000]
    test_x = x[-1000:]

    # train models
    print("Training models")
    train_sizes = [1, 2, 4, 8, 16, 32, 64]
    test_sizes = [6, 24, 96]

    def get_models(sizes, save_path):
        if os.path.exists(save_path):
            return pkl.load(open(save_path, 'rb'))

        models = []
        for msize in tqdm(sizes):
            model = GaussianMixture(
                n_components=msize
            )
            model.fit(x)
            models.append(model)
        with open(save_path, 'wb') as of:
            pkl.dump(models, of)
        return models

    train_gen_models = get_models(train_sizes, 'train_gen_models.pkl')
    test_gen_models = get_models(test_sizes, 'test_gen_models.pkl')
    score_models = train_gen_models + test_gen_models
    #train_score_models = get_models(train_sizes, 'train_score_models.pkl')
    #test_score_models = get_models(test_sizes, 'test_score_models.pkl')

    #score_models = train_score_models + test_score_models

    print("Generating samples")
    train_samples = [model.sample(n_samples=10000)[0] for model in train_gen_models]
    test_samples = [model.sample(n_samples=10000)[0] for model in test_gen_models]

    print("Scoring log diffs")

    def build_meta_set(gen_models, score_models, samples):
        meta_x = [] # N x M x D
        meta_y = [] # N x M x M

        for model, samples in zip(gen_models, samples):
            y = [model.score_samples(samples) - smodel.score_samples(samples) for smodel in score_models]
            y = np.stack(y, axis=-1)
            meta_x.append(samples)
            meta_y.append(y)

        meta_x = np.stack(meta_x, axis=1)
        meta_y = np.stack(meta_y, axis=-1)

        return meta_x, meta_y

    meta_train_x, meta_train_y = build_meta_set(train_gen_models, score_models, train_samples)
    meta_test_x, meta_test_y = build_meta_set(test_gen_models, score_models, test_samples)

    np.save('meta_train.npy', {'x': meta_train_x, 'y': meta_train_y})
    np.save('meta_test.npy', {'x': meta_test_x, 'y': meta_test_y})


