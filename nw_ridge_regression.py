from scipy.optimize import minimize
from scipy.stats import uniform
from typing import Callable, Union
import numpy as np
from scipy.sparse import issparse, csr_matrix
from concurrent.futures import ThreadPoolExecutor

def gaussian_kernel(X1: np.ndarray, X2: np.ndarray, h: float) -> np.ndarray:
    """
    Compute the Gaussian kernel between two matrices X1 and X2.
    
    Parameters:
        X1, X2: np.ndarray
            Input matrices.
        h: float
            Bandwidth parameter for the Gaussian kernel.
    
    Returns:
        np.ndarray: Kernel matrix
    """
    if issparse(X1) or issparse(X2):
        X1, X2 = csr_matrix(X1), csr_matrix(X2)
        norm = lambda x: x.power(2).sum(axis=1)
        K = -0.5 * (norm(X1) - 2 * X1.dot(X2.T) + norm(X2).T)
    else:
        pairwise_diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        squared_diff = np.square(pairwise_diff)
        K = -0.5 * np.sum(squared_diff, axis=-1)
    
    return np.exp(K / h**2)

def adamW_optimizer(
        loss_func: Callable[[np.ndarray], float], 
        initial_params: np.ndarray, 
        lr: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        epsilon: float = 1e-8, 
        weight_decay: float = 0.01, 
        max_iter: int = 1
    ) -> np.ndarray:
    """
    Optimize a function using the AdamW optimization algorithm.
    
    Parameters:
        loss_func: Callable
            The loss function to be optimized.
        initial_params: np.ndarray
            Initial parameters for the optimization.
        lr: float
            Learning rate.
        beta1, beta2: float
            Exponential decay rates for moment estimates.
        epsilon: float
            Small constant to prevent division by zero.
        weight_decay: float
            Weight decay parameter.
        max_iter: int
            Maximum number of iterations for the optimizer.
    
    Returns:
        np.ndarray: Optimized parameters
    """
    m = np.zeros_like(initial_params)
    v = np.zeros_like(initial_params)
    t = 0

    params = initial_params.copy()

    for _ in range(max_iter):
        grad = loss_func(params)
        t += 1

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        params -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * params)

    return params
def _to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return X
    
class NadarayaWatsonRidgeRegression:
    """
    Integrated ridge regression model with Nadaraya-Watson kernel smoothing.
    """
    def __init__(
            self, 
            alpha: float = 1.0, 
            h: float = 0.5, 
            kernel_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = gaussian_kernel
        ):
        self.alpha = alpha  # Ridge regularization parameter
        self.h = h  # Bandwidth parameter for the kernel
        self.kernel_func = kernel_func  # Kernel function
        self.w = None  # Ridge regression weights
        self.b = None  # Ridge regression bias term
        self.X_train = None  # Training features
        self.y_train = None  # Training labels
        self.kernel_cache = {}  # Cache for kernel values

    def _clear_kernel_cache(self):
        """Clear the kernel cache to free memory."""
        self.kernel_cache.clear()

    def _compute_kernel_chunk(self, X_chunk: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        """
        Compute and cache kernel values for a chunk of data.
        
        Parameters:
            X_chunk: np.ndarray
                A subset of the training data.
            X_query: np.ndarray
                The query points for which to compute the kernel.
                
        Returns:
            np.ndarray: The computed kernel values.
        """
        key = (tuple(X_chunk.flatten()), tuple(X_query.flatten()))
        if key in self.kernel_cache:
            return self.kernel_cache[key]

        kernel_values = self.kernel_func(X_chunk, X_query, self.h)
        self.kernel_cache[key] = kernel_values

        return kernel_values

    def fit(self, X: Union[np.ndarray, csr_matrix, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Fit the model to the given training data.
        
        Parameters:
            X: Union[np.ndarray, csr_matrix]
                Feature matrix.
            y: np.ndarray
                Target vector.
        """
        X = _to_numpy(X) 
        y = _to_numpy(y)
        # Clear any previous kernel cache
        self._clear_kernel_cache()

        # Convert to CSR format if the input is sparse
        if issparse(X):
            X = csr_matrix(X)


        self.X_train = X
        self.y_train = y
        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)

        def loss(params: np.ndarray) -> float:
            """
            Custom loss function for AdamW optimization.
            
            Parameters:
                params: np.ndarray
                    Model parameters including weights and bias.
                    
            Returns:
                float: The computed loss value.
            """
            w, b = params[:-1], params[-1]
            linear_preds = X.dot(w) + b
            kernel_corrections = self._nadaraya_watson(X, y - linear_preds, X)
            return np.sum((y - (linear_preds + kernel_corrections)) ** 2) + self.alpha * np.sum(w ** 2)

        optimal_params = adamW_optimizer(loss, initial_params)
        self.w, self.b = optimal_params[:-1], optimal_params[-1]

    def predict(self, X: Union[np.ndarray, csr_matrix, pd.DataFrame]) -> np.ndarray:
        """
        Predict the target values for a new set of data points.
        
        Parameters:
            X: Union[np.ndarray, csr_matrix]
                Feature matrix for the data points to predict.
                
        Returns:
            np.ndarray: The predicted target values.
        """
        X = _to_numpy(X) 

        # Convert to CSR format if the input is sparse
        if issparse(X):
            X = csr_matrix(X)
        
        linear_preds = X.dot(self.w) + self.b
        kernel_corrections = self._nadaraya_watson(self.X_train, self.y_train - (self.X_train.dot(self.w) + self.b), X)
        
        return linear_preds + kernel_corrections

    def _nadaraya_watson(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], X_query: Union[np.ndarray, pd.DataFrame], batch_size: int = 50) -> np.ndarray:

            """
            Compute the Nadaraya-Watson estimator for a set of query points.
            
            Parameters:
                X: np.ndarray
                    Feature matrix for the training data.
                y: np.ndarray
                    Target vector for the training data.
                X_query: np.ndarray
                    Feature matrix for the query points.
                batch_size: int
                    Size of the batch for batch processing.
                    
            Returns:
                np.ndarray: The Nadaraya-Watson estimates for the query points.
            """
            X = _to_numpy(X) 
            y = _to_numpy(y)  
            X_query = _to_numpy(X_query) 
            n_batches = (X.shape[0] + batch_size - 1) // batch_size
            weighted_sums = np.zeros(X_query.shape[0])
            total_weights = np.zeros(X_query.shape[0])

            def process_batch(i: int) -> tuple:
                """
                Process a batch of data and compute weighted sums and weights for Nadaraya-Watson estimation.
                
                Parameters:
                    i: int
                        Batch index.
                        
                Returns:
                    tuple: Weighted sums and total weights for the batch.
                """
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.shape[0])
                X_chunk = X[start_idx:end_idx]
                y_chunk = y[start_idx:end_idx]
                weights = self._compute_kernel_chunk(X_chunk, X_query)

                weighted_sums_local = np.sum(weights * y_chunk[:, np.newaxis], axis=0)
                total_weights_local = np.sum(weights, axis=0)

                return weighted_sums_local, total_weights_local

            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, range(n_batches)))

            for weighted_sums_batch, total_weights_batch in batch_results:
                weighted_sums += weighted_sums_batch
                total_weights += total_weights_batch

            # Handle the case where total_weights is zero to avoid division by zero
            return np.divide(weighted_sums, total_weights, out=np.zeros_like(weighted_sums), where=total_weights != 0)

