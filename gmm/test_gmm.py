import numpy as np
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
    means = np.random.uniform(-4, 4, (n_components, n_features))

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
    data_path = 'data.npy'
    if not os.path.exists(data_path):
        print("Generating data")
        max_components = 64
        logn_components = int(np.log2(max_components))
        data = []

        for i in range(0, logn_components + 1):
            X, y = gen_gmm_data(
                n_samples=1000,
                n_components=2**i,
                n_features=32,
                seed=0,
            )
            data.append(X)

        data = np.stack(data, axis=0)
        np.save(data_path, data)

    else:
        print("Loading data")
        data = np.load(data_path)
        logn_components = len(data)

    # train models
    print("Training models")
    models = []
    for i, idata in enumerate(data):
        models.append([])
        for j in range(logn_components):
            model = GaussianMixture(
                n_components=2**j
            )
            model.fit(idata)
            models[-1].append(model)

    print("Training models2")
    models2 = []
    for i, idata in enumerate(data):
        models2.append([])
        for j in range(logn_components):
            model = GaussianMixture(
                n_components=2**j
            )
            model.fit(idata)
            models2[-1].append(model)

    print("Generating samples")
    samples = [[model.sample(n_samples=10000)[0] for model in model_l] for model_l in models]
    test_samples = [[model.sample(n_samples=1000)[0] for model in model_l] for model_l in models]
    samples2 = [[model.sample(n_samples=10000)[0] for model in model_l] for model_l in models2]

    print("Plotting regret")
    for idata in list(range(logn_components)):
        fig, axs = plt.subplots(
            ncols=logn_components, nrows=logn_components,
            figsize=(25, 15), layout='constrained'
        )
        plt.subplots_adjust(left=0.04, top=0.07)

        for col in range(logn_components):
            ax = axs[0, col]
            ax.set_title(f'{2**col}', pad=20, fontsize=30)

        for row in range(logn_components):
            ax = axs[row, 0]
            ax.set_ylabel(f'{2**row}', labelpad=20, fontsize=30)

        kl_diffs = [[None for _ in range(logn_components)] for _ in range(logn_components)]

        for imodel_idx in range(logn_components):
            for jmodel_idx in range(logn_components):
                imodel = models[idata][imodel_idx]
                jmodel = models2[idata][jmodel_idx]

                # forward KL
                iregret = imodel.score_samples(samples[idata][imodel_idx]) - jmodel.score_samples(samples2[idata][imodel_idx])

                # reverse KL
                jregret = jmodel.score_samples(samples2[idata][jmodel_idx]) - imodel.score_samples(samples[idata][jmodel_idx])

                iregret = iregret.reshape(10, 1000).mean(0)
                jregret = jregret.reshape(10, 1000).mean(0)

                kl_diffs[imodel_idx][jmodel_idx] = iregret - jregret

        for imodel_idx in range(logn_components):
            imodel_std = np.std(kl_diffs[imodel_idx][imodel_idx])
            kl_diffs[imodel_idx] /= imodel_std
            for jmodel_idx in range(logn_components):
                sns.kdeplot(kl_diffs[imodel_idx][jmodel_idx], ax=axs[imodel_idx][jmodel_idx], bw_adjust=0.5)
                axs[imodel_idx][jmodel_idx].axvline(x=0, color='black', linestyle=':', alpha=0.5)
                axs[imodel_idx][jmodel_idx].axvline(x=kl_diffs[imodel_idx][jmodel_idx].mean(), color='blue', alpha=0.5)

                """
                squash_data = np.concatenate((
                    samples[idata][imodel_idx],
                    samples[idata][jmodel_idx],
                ), axis=-1)

                squash_test_data = np.concatenate((
                    test_samples[idata][imodel_idx],
                    test_samples[idata][jmodel_idx],
                ), axis=-1)

                reg = LinearRegression().fit(squash_data, iregret - jregret)
                test_pred = reg.predict(squash_test_data)
                loss = (test_pred - (test_iregret - test_jregret))**2
                import ipdb; ipdb.set_trace()
                """

        fig.savefig(f'{idata}_plus.png')


    """
    # Print some basic statistics
    print(f"Generated {len(X)} samples from {len(np.unique(y))} components")
    print(f"Shape of data: {X.shape}")

    # Optional: Plot the results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.title('Generated GMM Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    except ImportError:
        pass
    """

