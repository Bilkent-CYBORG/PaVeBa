import numpy as np


class IndependentSubgaussianModel:
    def __init__(
        self, input_dim, output_dim, noise_var, kernel, delta, design_count=206, conf_contraction=1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.design_count = design_count
        self.conf_contraction = conf_contraction

        # Data containers.
        self.clear_data()

        self.delta = delta
        self.noise_var = noise_var

        self.kernel = kernel

        self.model = None

    def add_sample(self, X_t, Y_t):
        # Last column of X_t are sample space indices.
        for idx, y in zip(X_t[..., -1].astype(int), Y_t):
            self.design_samples[idx] = np.concatenate(
                [self.design_samples[idx], y.reshape(-1, self.output_dim)], 0
            )

    def clear_data(self):
        self.design_samples = [np.empty((0, self.output_dim)) for _ in range(self.design_count)]

    def update(self):
        self.means = [
            np.mean(design, axis=0) if len(design) > 0 else 0 for design in self.design_samples
        ]
        self.variances = []
        for design in self.design_samples:
            ni = max(1, len(design))

            # Original radius
            t1 = (8 * self.noise_var / ni)
            t2 = np.log(  # ni**2 is equal to t**2 since only active arms are sampled
                (np.pi**2 * (self.output_dim + 1) * self.design_count * ni**2) / (6 * self.delta)
            )
            r = np.sqrt(t1 * t2)

            self.variances.append(r / self.conf_contraction)

    def train(self):
        pass

    def predict(self, test_X):
        indices = test_X[..., -1].astype(int)
        means = [self.means[ind] for ind in indices]
        variances = [self.variances[ind] for ind in indices]

        return means, variances
