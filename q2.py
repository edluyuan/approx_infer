import numpy as np
import matplotlib.pyplot as plt

# Constants
DATA_FILE = 'co2.txt'
BASE_YEAR = 1980.0  # Base year for adjusting the time variable

# Prior Parameters
PRIOR_MEAN = np.array([0.0, 360.0])  # [a, b]
PRIOR_COVARIANCE = np.diag([10.0**2, 100.0**2])  # Covariance matrix
PRIOR_COV_INV = np.linalg.inv(PRIOR_COVARIANCE)

# Initialize lists to store time and CO2 measurements
time = []
co2_average = []

# Load data from the file
try:
    with open(DATA_FILE, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue

            # Split the line into columns
            columns = line.split()

            # Ensure there are at least 5 columns
            if len(columns) >= 5:
                try:
                    decimal_year = float(columns[2])
                    average_co2 = float(columns[3])

                    # Adjust time relative to the base year
                    adjusted_time = decimal_year - BASE_YEAR
                    time.append(adjusted_time)
                    co2_average.append(average_co2)
                except ValueError:
                    # Handle lines with non-numeric data
                    continue
except FileNotFoundError:
    raise FileNotFoundError(f"The file {DATA_FILE} was not found.")

# Convert lists to NumPy arrays for efficient computations
t = np.array(time)  # Time variable (years since BASE_YEAR)
y = np.array(co2_average)  # Average CO2 measurements

# Design Matrix Construction
# Each row corresponds to [t_i, 1] representing the linear model coefficients
X = np.column_stack((t, np.ones_like(t)))

# Posterior Covariance Calculation
# Σ_post = (Σ_prior^{-1} + X^T X)^{-1}
posterior_cov_inv = PRIOR_COV_INV + X.T @ X
posterior_covariance = np.linalg.inv(posterior_cov_inv)

# Posterior Mean Calculation
# μ_post = Σ_post (Σ_prior^{-1} μ_prior + X^T y)
posterior_mean = posterior_covariance @ (PRIOR_COV_INV @ PRIOR_MEAN + X.T @ y)

# Extract posterior estimates for parameters a and b
a_posterior, b_posterior = posterior_mean

# Display the results
print("Posterior Mean of a (slope):", a_posterior)
print("Posterior Mean of b (intercept):", b_posterior)
print("Posterior Covariance Matrix:\n", posterior_covariance)

# Optional: Visualization of the Data and Posterior Mean Line
plt.figure(figsize=(10, 6))
plt.scatter(t + BASE_YEAR, y, label='CO2 Measurements', alpha=0.5)
plt.plot(t + BASE_YEAR, a_posterior * t + b_posterior, color='red', label='Posterior Mean')
plt.xlabel('Year')
plt.ylabel('Average CO2')
plt.title('CO2 Levels Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import scipy.stats as stats

# Assumptions:
# t: numpy array of shape (N,) containing times (decimal years)
# y: numpy array of shape (N,) containing observed CO2 concentrations (fobs(t))
# a_MAP, b_MAP: scalars for MAP estimates of a and b, obtained from previous steps

# Compute residuals: gobs(t) = fobs(t) - (a_MAP * t + b_MAP)
gobs = y - (a_posterior * t + b_posterior)

# Plot residuals against time as a line plot
plt.figure(figsize=(10, 6))
plt.plot(t, gobs, label='Residuals', color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Time (decimal years)')
plt.ylabel('Residual $g_{obs}(t)$ (ppm)')
plt.title('Residuals vs. Time')
plt.legend()
plt.grid(True)
plt.savefig(fname="2b1.png", dpi=300)
plt.show()

# Print mean and variance of residuals
mean_res = np.mean(gobs)
var_res = np.var(gobs, ddof=1)  # sample variance
print(f"Mean of residuals: {mean_res:.4f}")
print(f"Variance of residuals: {var_res:.4f}")

# Test normality (e.g., Shapiro-Wilk test)
shapiro_stat, shapiro_p = stats.shapiro(gobs)
print(f"Shapiro-Wilk test statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")

# Plot histogram of residuals and compare to a standard normal PDF
plt.figure(figsize=(10, 5))
plt.hist(gobs, bins=30, density=True, alpha=0.7, edgecolor='black', label='Residual Histogram')
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Density')

# Plot a standard normal PDF to reflect the prior belief ε(t) ~ N(0,1)
xvals = np.linspace(mean_res - 4*np.sqrt(var_res), mean_res + 4*np.sqrt(var_res), 200)
normal_pdf = stats.norm.pdf(xvals, loc=0, scale=1)  # N(0,1)
plt.plot(xvals, normal_pdf, 'r--', label='N(0,1) PDF (prior)')
plt.legend()
plt.grid(True)
plt.savefig(fname="2b2.png", dpi=300)
plt.show()

# Q-Q plot to check normality
plt.figure(figsize=(10,5))
stats.probplot(gobs, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.grid(True)
plt.savefig(fname="2b3.png", dpi=300)
plt.show()


import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from decimal import Decimal
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod

# ------------------------------
# Define Kernel and Parameters
# ------------------------------

@dataclass
class KernelParameters(ABC):
    """Abstract class for kernel parameters."""
    pass

class Kernel(ABC):
    """Abstract base class for kernels."""
    Parameters: KernelParameters = None

    @abstractmethod
    def _kernel(
        self, parameters: KernelParameters, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the kernel between a single pair of points x and y.
        """
        raise NotImplementedError

    def kernel(
        self, parameters: KernelParameters, x: jnp.ndarray, y: jnp.ndarray = None
    ) -> jnp.ndarray:
        # If y is None, set y = x for k(x,x)
        if y is None:
            y = x
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        assert x.shape[1] == y.shape[1], "Dimension mismatch in kernel inputs."

        # Use vmap to efficiently compute gram matrix
        return jax.vmap(
            lambda x_i: jax.vmap(lambda y_i: self._kernel(parameters, x_i, y_i))(y)
        )(x)

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray = None, **parameter_args
    ) -> jnp.ndarray:
        parameters = self.Parameters(**parameter_args)
        return self.kernel(parameters, x, y)

@dataclass
class CombinedKernelParameters(KernelParameters):
    """
    Parameters for the combined kernel:
    k(s, t) = theta^2 * exp(-2 sin^2(pi(s-t)/tau)/sigma^2)
              + phi^2 * exp(-(s-t)^2/(2 * eta^2))
              + zeta^2 delta(s=t)
    All parameters are stored as logs.
    """
    log_theta: float
    log_sigma: float
    log_phi: float
    log_eta: float
    log_tau: float
    log_zeta: float

    @property
    def theta(self) -> float:
        return jnp.exp(self.log_theta)

    @property
    def sigma(self) -> float:
        return jnp.exp(self.log_sigma)

    @property
    def phi(self) -> float:
        return jnp.exp(self.log_phi)

    @property
    def eta(self) -> float:
        return jnp.exp(self.log_eta)

    @property
    def tau(self) -> float:
        return jnp.exp(self.log_tau)

    @property
    def zeta(self) -> float:
        return jnp.exp(self.log_zeta)

class CombinedKernel(Kernel):
    Parameters = CombinedKernelParameters

    def _kernel(
        self,
        parameters: CombinedKernelParameters,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        dist = jnp.pi * (x - y) / parameters.tau
        periodic_part = parameters.theta**2 * jnp.exp(-2 * (jnp.sin(dist)**2) / parameters.sigma**2)
        se_part = parameters.phi**2 * jnp.exp(-((x - y)**2) / (2 * parameters.eta**2))
        noise_part = parameters.zeta**2 * jnp.where(jnp.allclose(x, y), 1.0, 0.0)
        return periodic_part + se_part + noise_part

# ------------------------------
# Gaussian Process and Parameters
# ------------------------------

@dataclass
class GaussianProcessParameters:
    """
    Parameters for a Gaussian Process:
    log_sigma: noise parameter (log)
    kernel: dictionary containing kernel params
    """
    log_sigma: float
    kernel: Dict[str, Any]

    @property
    def sigma(self) -> float:
        return jnp.exp(self.log_sigma)

    @property
    def variance(self) -> float:
        return self.sigma**2

class GaussianProcess:
    def __init__(self, kernel: Kernel, x: np.ndarray, y: np.ndarray):
        """
        Initialize GP with training data.
        x: (N, D)
        y: (N,)
        """
        self.number_of_train_points = x.shape[0]
        self.x = jnp.array(x)
        self.y = jnp.array(y)
        self.kernel = kernel

    def _compute_kxx_shifted_cholesky_decomposition(self, parameters: GaussianProcessParameters):
        kxx = self.kernel(self.x, **parameters.kernel)
        kxx_shifted = kxx + parameters.variance * jnp.eye(self.number_of_train_points)
        kxx_shifted_cholesky_decomposition, lower_flag = jax.scipy.linalg.cho_factor(
            a=kxx_shifted, lower=True
        )
        return kxx_shifted_cholesky_decomposition, lower_flag

    def posterior_distribution(self, x: jnp.ndarray, **parameter_args) -> Tuple[jnp.ndarray, jnp.ndarray]:
        parameters = GaussianProcessParameters(**parameter_args)
        kxy = self.kernel(self.x, x, **parameters.kernel)
        kyy = self.kernel(x, **parameters.kernel)
        kxx_shifted_cholesky_decomposition, lower_flag = self._compute_kxx_shifted_cholesky_decomposition(parameters)

        alpha = jax.scipy.linalg.cho_solve((kxx_shifted_cholesky_decomposition, lower_flag), self.y)
        mean = (kxy.T @ alpha).reshape(-1)

        v = jax.scipy.linalg.cho_solve((kxx_shifted_cholesky_decomposition, lower_flag), kxy)
        covariance = kyy - kxy.T @ v
        return mean, covariance

    def posterior_negative_log_likelihood(self, **parameter_args) -> jnp.float64:
        parameters = GaussianProcessParameters(**parameter_args)
        kxx_shifted_cholesky_decomposition, lower_flag = self._compute_kxx_shifted_cholesky_decomposition(parameters)
        alpha = jax.scipy.linalg.cho_solve((kxx_shifted_cholesky_decomposition, lower_flag), self.y)
        N = self.number_of_train_points
        neg_log_likelihood = 0.5 * (self.y.T @ alpha) + jnp.sum(jnp.log(jnp.diag(kxx_shifted_cholesky_decomposition))) + (N / 2)*jnp.log(2*jnp.pi)
        return neg_log_likelihood

    def _compute_gradient(self, **parameter_args) -> Dict[str, Any]:
        gradients = jax.grad(lambda p: self.posterior_negative_log_likelihood(**p))(parameter_args)
        return gradients

    def train(self, optimizer: optax.GradientTransformation, number_of_training_iterations: int, **parameter_args) -> GaussianProcessParameters:
        opt_state = optimizer.init(parameter_args)
        for _ in range(number_of_training_iterations):
            gradients = self._compute_gradient(**parameter_args)
            updates, opt_state = optimizer.update(gradients, opt_state)
            parameter_args = optax.apply_updates(parameter_args, updates)
        return GaussianProcessParameters(**parameter_args)

# ------------------------------
# Bayesian Linear Regression Posterior
# ------------------------------

@dataclass
class LinearRegressionParameters:
    mean: np.ndarray
    covariance: np.ndarray

    @property
    def precision(self) -> np.ndarray:
        return np.linalg.inv(self.covariance)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.mean.T @ x)

@dataclass
class Theta:
    linear_regression_parameters: LinearRegressionParameters
    sigma: float

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def precision(self) -> float:
        return 1.0 / self.variance

def compute_linear_regression_posterior(
    x: np.ndarray,
    y: np.ndarray,
    prior_linear_regression_parameters: LinearRegressionParameters,
    residuals_precision: float,
) -> LinearRegressionParameters:
    # Posterior covariance and mean for Bayesian linear regression
    Sigma0_inv = np.linalg.inv(prior_linear_regression_parameters.covariance)
    A = Sigma0_inv + residuals_precision * (x @ x.T)
    Sigma_post = np.linalg.inv(A)
    mu_post = Sigma_post @ (Sigma0_inv @ prior_linear_regression_parameters.mean + residuals_precision * x @ y)
    return LinearRegressionParameters(mean=mu_post, covariance=Sigma_post)

# ------------------------------
# Helper Functions
# ------------------------------

def construct_design_matrix(t: np.ndarray) -> np.ndarray:
    """Design matrix for linear regression: [t; 1]. Shape: (2, N)."""
    return np.stack((t, np.ones_like(t)), axis=1).T

# ------------------------------
# The main function f
# ------------------------------
def f(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    min_year: float,
    prior_linear_regression_parameters: LinearRegressionParameters,
    linear_regression_sigma: float,
    kernel: CombinedKernel,
    gaussian_process_parameters: GaussianProcessParameters,
    learning_rate: float,
    number_of_iterations: int,
    save_path: str,
) -> None:
    # Bayesian linear regression posterior
    x_train = construct_design_matrix(t_train)
    prior_theta = Theta(
        linear_regression_parameters=prior_linear_regression_parameters,
        sigma=linear_regression_sigma,
    )
    posterior_linear_regression_parameters = compute_linear_regression_posterior(
        x_train,
        y_train,
        prior_linear_regression_parameters,
        residuals_precision=prior_theta.precision,
    )

    # Residuals
    residuals = y_train - posterior_linear_regression_parameters.predict(x_train)

    # Init Gaussian Process
    gaussian_process = GaussianProcess(
        kernel, t_train.reshape(-1, 1), residuals.reshape(-1)
    )

    # Predictions before training hyperparameters
    x_test_dm = construct_design_matrix(t_test)
    linear_prediction = posterior_linear_regression_parameters.predict(x_test_dm).reshape(-1)
    mean_prediction, covariance_prediction = gaussian_process.posterior_distribution(
        t_test.reshape(-1, 1), **asdict(gaussian_process_parameters)
    )

    # Plot untrained
    std_dev = np.sqrt(np.diag(np.array(covariance_prediction)))
    plt.figure(figsize=(7,7))
    plt.scatter(t_train + min_year, y_train, s=2, color='blue', label='historical data')
    plt.plot(t_test + min_year, linear_prediction + np.array(mean_prediction), color='gray', label='prediction (untrained)')
    plt.fill_between(
        t_test + min_year,
        linear_prediction + mean_prediction - std_dev,
        linear_prediction + mean_prediction + std_dev,
        facecolor=(0.8,0.8,0.8),
        label='error bound (one stdev)'
    )
    plt.xlabel("date (decimal year)")
    plt.ylabel("parts per million")
    plt.title("Global Mean CO2 Concentration Prediction (Untrained Hyperparameters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "-extrapolation-untrained.png", bbox_inches='tight')
    plt.close()

    # Print parameters before training
    param_dict_before = {k.strip("log_"): "%.2E" % Decimal(np.exp(v)) for k,v in gaussian_process_parameters.kernel.items()}
    param_dict_before["sigma"] = "%.2E" % Decimal(float(gaussian_process_parameters.sigma))
    df_parameters = pd.DataFrame(list(param_dict_before.items()), columns=["parameter","value"]).set_index("parameter").sort_values("parameter")
    print("Untrained Parameters:")
    print(df_parameters)
    df_parameters.to_csv(save_path + "-untrained-parameters.csv")

    # Train GP hyperparameters
    optimizer = optax.adam(learning_rate)
    gp_params_trained = gaussian_process.train(optimizer, number_of_iterations, **asdict(gaussian_process_parameters))

    # Print parameters after training
    param_dict_after = {k.strip("log_"): "%.2E" % Decimal(np.exp(v)) for k,v in gp_params_trained.kernel.items()}
    param_dict_after["sigma"] = "%.2E" % Decimal(float(gp_params_trained.sigma))
    df_parameters = pd.DataFrame(list(param_dict_after.items()), columns=["parameter","value"]).set_index("parameter").sort_values("parameter")
    print("Trained Parameters:")
    print(df_parameters)
    df_parameters.to_csv(save_path + "-trained-parameters.csv")

    # Predictions after training hyperparameters
    mean_prediction, covariance_prediction = gaussian_process.posterior_distribution(
        t_test.reshape(-1, 1), **asdict(gp_params_trained)
    )
    std_dev = np.sqrt(np.diag(np.array(covariance_prediction)))
    linear_prediction = posterior_linear_regression_parameters.predict(x_test_dm).reshape(-1)

    # Plot trained
    plt.figure(figsize=(7,7))
    plt.scatter(t_train + min_year, y_train, s=2, color='blue', label='historical data')
    plt.plot(t_test + min_year, linear_prediction + mean_prediction, color='gray', label='prediction (trained)')
    plt.fill_between(
        t_test + min_year,
        linear_prediction + mean_prediction - std_dev,
        linear_prediction + mean_prediction + std_dev,
        facecolor=(0.8,0.8,0.8),
        label='error bound (one stdev)'
    )
    plt.xlabel("date (decimal year)")
    plt.ylabel("parts per million")
    plt.title("Global Mean CO2 Concentration Prediction (Trained Hyperparameters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "-extrapolation-trained.png", bbox_inches='tight')
    plt.close()

# ------------------------------
# Example usage (commented out)
# ------------------------------
# Suppose we have:
# t_train, y_train: training data arrays
# t_test: test times for extrapolation
# min_year: offset to display correct decimal years on plots
# prior_parameters = LinearRegressionParameters(mean=np.array([0.0,360.0]), covariance=np.diag([100,10000]))
# gp_parameters = GaussianProcessParameters(
#     log_sigma=jnp.log(0.1),
#     kernel={
#         "log_theta": jnp.log(1.0),
#         "log_sigma": jnp.log(0.5),
#         "log_phi": jnp.log(0.5),
#         "log_eta": jnp.log(2.0),
#         "log_tau": jnp.log(1.0),
#         "log_zeta": jnp.log(0.1)
#     }
# )
# kernel = CombinedKernel()
# f(t_train, y_train, t_test, min_year, prior_parameters, 1.0, kernel, gp_parameters, 0.01, 50, "results")


import numpy as np
import pandas as pd

# ---------------------------------------
# Load and prepare data
# ---------------------------------------
co2_data = []
with open('co2.txt', 'r') as f:
    for line in f:
        line=line.strip()
        if line.startswith('#') or len(line)==0:
            continue
        parts=line.split()
        if len(parts)>=5:
            decimal_year = float(parts[2])
            average_co2 = float(parts[3])
            co2_data.append((decimal_year, average_co2))

co2_data = np.array(co2_data) # shape (N, 2)
t = co2_data[:,0]
y = co2_data[:,1]

# Define training cutoff
# According to the prompt: Observe data up to Sept 2007 => decimal ~ 2007.708
cutoff = 2007.708
train_mask = t <= cutoff
test_mask = t >= 2007.708

t_train = t[train_mask]
y_train = y[train_mask]

# We want monthly intervals between Sept 2007 and Dec 2020
# Sept 2007 decimal year ~ 2007.708
# Dec 2020 decimal year ~ 2020.958
t_test = []
start_year = 2007.708
end_year = 2020.958
current = start_year
while current <= end_year:
    t_test.append(current)
    current += 1/12.0
t_test = np.array(t_test)

# ---------------------------------------
# Suppose from earlier Bayesian linear regression steps, you have:
a_MAP = 1.82     # Example value, replace with your computed MAP estimate
b_MAP = 338.0    # Example value, replace with your computed MAP estimate

# Construct the prior parameters for linear regression
# Suppose your prior was a ~ N(0,10^2) and b ~ N(360,100^2)
mu0 = np.array([0.0, 360.0])
Sigma0 = np.diag([10.0**2, 100.0**2])

from dataclasses import asdict
import jax.numpy as jnp
from decimal import Decimal

# ---------------------------------------
# Use the classes and functions from the previous provided code
# (We assume you've copied the entire code block with all classes and the function f into the same script.)
# That block defined f(...), CombinedKernel, GaussianProcessParameters, LinearRegressionParameters, etc.

# Initialize linear regression parameters
#from jax.config import config
#config.update("jax_enable_x64", True)  # ensure double precision if needed

prior_linear_regression_parameters = LinearRegressionParameters(mean=mu0, covariance=Sigma0)
linear_regression_sigma = 1.0  # as assumed before (noise std dev = 1)

# Initialize kernel
kernel = CombinedKernel()

# Choose GP parameters (log space)
# These are guesses; you should tune them based on your data analysis.
gp_params = GaussianProcessParameters(
    log_sigma=jnp.log(0.1),  # noise parameter sigma for GP
    kernel={
        "log_theta": jnp.log(1.0),
        "log_sigma": jnp.log(0.5),
        "log_phi": jnp.log(0.5),
        "log_eta": jnp.log(2.0),
        "log_tau": jnp.log(1.0),   # assuming 1-year periodic
        "log_zeta": jnp.log(0.1)
    }
)

# The min_year is used to adjust plotting. If you want to plot actual decimal years on x-axis:
min_year = 0.0  # If you want to see actual decimal years, just set min_year=0.0 and you'll see full decimal years.

# Learning rate and number of iterations for GP hyperparameter training
learning_rate = 0.01
number_of_iterations = 50

# Save path prefix
save_path = "gp_co2_extrapolation"

# ---------------------------------------
# Run the function f:
# This will:
# 1) Fit the linear regression posterior to training data.
# 2) Compute residuals.
# 3) Fit and train a GP to the residuals.
# 4) Produce plots and parameter tables.

f(
    t_train=t_train,
    y_train=y_train,
    t_test=t_test,
    min_year=min_year,
    prior_linear_regression_parameters=prior_linear_regression_parameters,
    linear_regression_sigma=linear_regression_sigma,
    kernel=kernel,
    gaussian_process_parameters=gp_params,
    learning_rate=learning_rate,
    number_of_iterations=number_of_iterations,
    save_path=save_path,
)

# ---------------------------------------
# After running this, check the generated plots:
# - gp_co2_extrapolation-extrapolation-untrained.png: before GP parameter optimization
# - gp_co2_extrapolation-untrained-parameters.csv: initial parameters
# - gp_co2_extrapolation-extrapolation-trained.png: after GP parameter optimization
# - gp_co2_extrapolation-trained-parameters.csv: tuned parameters
#
# Inspect the behavior of the extrapolation:
# - Does it maintain periodic/seasonal structure?
# - Does the variance grow?
# - How does changing log_eta, log_tau, log_zeta, etc. affect the extrapolation?
