import jax
import jax.numpy as jnp
import optax
from tqdm.auto import trange
# --- Step 1: Load CO2 data (same as before, using base year 1980) ---
t = []
y = []
with open('co2.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
            continue
        parts = line.split()
        if len(parts) >= 5:
            decimal_year = float(parts[2])
            average_co2 = float(parts[3])
            # Shift base year to 1980
            decimal_year -= 1980.0
            t.append(decimal_year)
            y.append(average_co2)

t = jnp.array(t)
y = jnp.array(y)


g_obs = y - (a_posterior * t + b_posterior)

# Training cutoff at 2007.708 => 27.708 after 1980
cutoff = 2007.708 - 1980.0
train_idx = t <= cutoff
t_train = t[train_idx]
g_train = g_obs[train_idx]


# --- Step 2: Define the kernel function ---
def kernel(x1, x2, theta, tau, sigma, phi, eta, zeta):
    dist = jnp.pi * (x1[:, None] - x2[None, :]) / tau
    periodic_part = theta**2 * jnp.exp(-2.0 * (jnp.sin(dist)**2) / sigma**2)
    se_part = (theta**2 * phi**2) * jnp.exp(-( (x1[:, None] - x2[None, :])**2 )/(2 * eta**2))

    # Check if x1 and x2 represent the same array
    # Ensure that shapes match and all elements are equal
    same_input = (x1.shape == x2.shape) and bool(jnp.all(x1 == x2))
    noise_part = zeta**2 * jnp.eye(len(x1)) if same_input else 0.0

    return periodic_part + se_part + noise_part

# --- Step 3: Negative log-likelihood ---
def neg_log_marginal_likelihood(params, X, y):
    # params is a dictionary of raw parameters in log-space
    # Exponentiate to ensure positivity
    theta = jnp.exp(params['log_theta'])
    sigma = jnp.exp(params['log_sigma'])
    phi = jnp.exp(params['log_phi'])
    eta = jnp.exp(params['log_eta'])
    zeta = jnp.exp(params['log_zeta'])
    tau = jnp.exp(params['log_tau'])
    # Construct K
    K = kernel(X, X, theta, tau, sigma, phi, eta, zeta)
    # Add a small jitter for numerical stability
    K += 1e-10 * jnp.eye(len(X))

    # Compute NLL = 0.5*y^T K^-1 y + 0.5 log|K| + n/2 log(2π)
    L = jnp.linalg.cholesky(K)
    # Solve for alpha = K^-1 y using Cholesky
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    n = len(X)
    nll = 0.5 * y.dot(alpha) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * n * jnp.log(2.0 * jnp.pi)
    return nll

# --- Step 4: Setup Optimization ---
# Initial guesses for parameters in log-space
init_params = {
    'log_theta': jnp.log(1.0),
    'log_sigma': jnp.log(1.0),
    'log_phi': jnp.log(1.0),
    'log_eta': jnp.log(5.0),
    'log_zeta': jnp.log(1.0),
    'log_tau': jnp.log(1.0)
}

# Use Adam optimizer from optax
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(init_params)

# We can jit the gradient function for efficiency
grad_fn = jax.grad(neg_log_marginal_likelihood)

# --- Step 5: Run Gradient-Based Optimization ---
num_iters = 200
params = init_params
for i in trange(num_iters, desc="Optimizing hyperparameters"):
    grads = grad_fn(params, t_train, g_train)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if i % 20 == 0:
        current_loss = neg_log_marginal_likelihood(params, t_train, g_train)
        print(f"Iteration {i}, NLL={current_loss}")

# After optimization
opt_theta = jnp.exp(params['log_theta'])
opt_sigma = jnp.exp(params['log_sigma'])
opt_phi = jnp.exp(params['log_phi'])
opt_eta = jnp.exp(params['log_eta'])
opt_zeta = jnp.exp(params['log_zeta'])
opt_tau = jnp.exp(params['log_tau'])

print("Optimized Hyperparameters:")
print("theta =", opt_theta, "tau =", opt_tau, "sigma =", opt_sigma, "phi =", opt_phi, "eta =", opt_eta, "zeta =", opt_zeta)

# --- Step 6: Make Predictions ---
# Suppose we want to predict beyond cutoff
test_start = 2007.708 - 1980.0
test_end = 2020.958 - 1980.0
num_months = int(round((test_end - test_start) * 12)) + 1
test_t = jnp.linspace(test_start, test_end, num_months)

K = kernel(t_train, t_train, opt_theta, opt_tau, opt_sigma, opt_phi, opt_eta, opt_zeta)
K_star = kernel(test_t, t_train, opt_theta, opt_tau, opt_sigma, opt_phi, opt_eta, opt_zeta)
K_star_star = kernel(test_t, test_t, opt_theta, opt_tau, opt_sigma, opt_phi, opt_eta, opt_zeta)

K_inv = jnp.linalg.inv(K)
g_mean = K_star @ K_inv @ g_train
g_cov = K_star_star - K_star @ K_inv @ K_star.T
g_std = jnp.sqrt(jnp.diag(g_cov))

f_pred = a_posterior * test_t + b_posterior + g_mean

# --- Step 7: Plot Results ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(t + 1980.0, y, 'k.', alpha=0.5, label='Observed CO2')
plt.axvspan(1980.0, 1980.0 + cutoff, color='gray', alpha=0.1, label='Training period')
plt.plot(test_t + 1980.0, f_pred, 'b-', label='Predicted mean')
plt.fill_between(test_t + 1980.0, f_pred - g_std, f_pred + g_std, color='blue', alpha=0.2, label='±1 std')

plt.title('GP Extrapolation of CO2 Concentrations with Optimized Hyperparameters')
plt.xlabel('Decimal Year')
plt.ylabel('CO2 (ppm)')
plt.grid(True)
plt.legend()
plt.savefig("extrapolation_optimized.png", dpi=300)
plt.show()
