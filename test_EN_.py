import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# # YOUR FUNCTION GOES HERE ---> The soft_threshold function is a utility function that performs element-wise
# soft-thresholding. It is used to update the coefficients in the L1-regularized term (Lasso) part of Elastic Net.
def soft_threshold(x, threshold):
    # The soft-threshold function, used in the coordinate gradient descent for L1 regularization.
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# The MyElasticNet function implements the coordinate gradient descent algorithm for Elastic Net regression. It takes
# the input data X and target variable y, along with the regularization parameter lam, the mixing parameter alpha,
# and the maximum number of iterations max_iter. It returns the final weight vector and the duality gap.
def MyElasticNet(X, y, lam=1.0, alpha=0.5, max_iter=1000):
    # Get the number of samples and features from the input data.
    n_samples, n_features = X.shape
    # Initialize the coefficient vector with zeros.
    betas = np.zeros(n_features)
    # Initialize the dual feasible vector with zeros.
    v = np.zeros(n_samples)
    # Initialize the duality gap with infinity.
    duality_gap = np.inf

    # Create a list to store the duality gap values at each iteration.
    duality_gap_track = []

    # Perform the coordinate gradient descent algorithm.
    for _ in range(max_iter):
        # Make a copy of the current coefficients to check for convergence later.
        betas_prev = betas.copy()
        # Loop through all features.
        for j in range(n_features):
            # Extract the j-th feature column from the input data.
            X_j = X[:, j]
            # Remove the j-th coefficient from the betas vector.
            betas_wo_j = np.delete(betas, j)
            # Remove the j-th feature column from the input data.
            X_wo_j = np.delete(X, j, axis=1)
            # Compute the residual without considering the j-th feature.
            r_j = y - X_wo_j @ betas_wo_j
            # Compute the inner product of the j-th feature and the residual.
            z_j = np.dot(X_j, r_j)
            # Update the j-th coefficient using soft-thresholding.
            betas[j] = soft_threshold(z_j, lam * alpha) / (np.sum(X_j**2) + lam * (1 - alpha))

        # Update the dual feasible vector.
        v = y - X @ betas
        # Normalize the dual feasible vector.
        v *= 1 / (1 + np.linalg.norm(v))
        # Compute the duality gap.
        duality_gap = 0.5 * np.sum((y - X @ betas)**2) + lam * alpha * np.sum(np.abs(betas)) + 0.5 * lam * (1 - alpha) * np.sum(betas**2) - 0.5 * np.sum(v**2)

        # Append the current duality gap value to the list.
        duality_gap_track.append(duality_gap)

        # Check for convergence: if the change in betas is smaller than a threshold, stop the iteration.
        if np.linalg.norm(betas - betas_prev) < 1e-4:
            break

    # Return the final coefficient vector and the duality gap.
    return betas, duality_gap, duality_gap_track

## <--- YOUR FUNCTION GOES HERE

## Small Problem

rs = 1603               # random seed
sample_size = 100       # number of samples
feature_size = 5        # number of features
noise_scale = 5.0       # standard deviation of noise 

# Generating the data set
X, y = make_friedman1(n_samples = sample_size,
                    n_features=feature_size,
                    noise=noise_scale, random_state=rs)

# Train - Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

## SCALING GOES HERE --->

####### SCALING #######
# Instantiate a StandardScaler object.
scaler = StandardScaler()
# Fit the scaler to the training data and transform it.
X_train = scaler.fit_transform(X_train)
# Transform the test data using the same scaler.
X_test = scaler.transform(X_test)

# Compute the mean of the training target variable.
y_train_mean = np.mean(y_train)
# Center the training target variable by subtracting the mean.
y_train = y_train - y_train_mean
# Center the test target variable by subtracting the training mean.
y_test = y_test - y_train_mean

## <--- SCALING GOES HERE

lam = 1.0               # lambda parameter
alp = 0.5               # alpha parameter
miter = 1000            # maximum number of iterations

# sklearn's implementation
EN = ElasticNet(alpha=lam, l1_ratio=alp).fit(X_train, y_train)     # sklearn's Elastic Net swaps alpha and lambda

# Your implementation
betabar, eta, duality_gap_track = MyElasticNet(X_train, y_train, lam=lam, alpha=alp, max_iter=miter)

print('Coefficients of sklearn implementation: ', EN.coef_)
print('Coefficients of your implementation: ', betabar)
print('The distance between two coefficients: ', np.linalg.norm(EN.coef_ - betabar))

print('Mean squared error: ', mean_squared_error(X_test @ betabar, y_test))
print('Duality Gap: ', eta)

# Plot the duality gap values.
plt.plot(duality_gap_track)
plt.xlabel('Iteration')
plt.ylabel('Duality Gap')
plt.title('Duality Gap Track')
plt.show()

###############################

## Large Problem

rs = 1603               # random seed
sample_size = 10000     # number of samples
feature_size = 50       # number of features
noise_scale = 5.0       # standard deviation of noise 

# Generating the data set
X, y = make_friedman1(n_samples = sample_size,
                    n_features=feature_size,
                    noise=noise_scale, random_state=rs)

# Train - Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rs)

## SCALING GOES HERE --->

####### SCALING #######

## <--- SCALING GOES HERE

lam = 1.0               # lambda parameter
alp = 0.5               # alpha parameter
miter = 1000            # maximum number of iterations

# sklearn's implementation
EN = ElasticNet(alpha=lam, l1_ratio=alp).fit(X_train, y_train)     # sklearn's Elastic Net swaps alpha and lambda

# Your implementation
betabar, eta = MyElasticNet(X_train, y_train, lam=lam, alpha=alp, max_iter=miter)

print('Coefficients of sklearn implementation: ', EN.coef_)
print('Coefficients of your implementation: ', betabar)
print('The distance between two coefficients: ', np.linalg.norm(EN.coef_ - betabar))

print('Mean squared error: ', mean_squared_error(X_test @ betabar, y_test))
print('Duality Gap: ', eta)
#%%
