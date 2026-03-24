import numpy as np

class OnlineRLS:
    """
    Recursive Least Squares for lightning-fast online linear regression.
    Updates weights in O(N^2) time without needing a learning rate.
    Ideal for in-loop latency prediction.
    """
    def __init__(self, num_features: int, forgetting_factor: float = 0.99, delta: float = 10.0):
        self.num_features = num_features
        self.lambda_factor = forgetting_factor
        
        # Weights vector (Theta)
        self.w = np.zeros(num_features)
        
        # Inverse covariance matrix (P) initialized to Identity * delta
        self.P = np.eye(num_features) * delta

    def predict(self, x: np.ndarray) -> float:
        """O(N) prediction."""
        # Ensure we don't return negative times for physical processes
        return max(0.0, float(np.dot(self.w, x)))

    def update(self, x: np.ndarray, y_true: float):
        """O(N^2) exact mathematical update."""
        # 1. Calculate gain vector K
        Px = np.dot(self.P, x)
        denominator = self.lambda_factor + np.dot(x, Px)
        
        # Prevent division by zero in degenerate cases
        if denominator == 0:
            return
            
        K = Px / denominator
        
        # 2. Calculate error
        error = y_true - np.dot(self.w, x)
        
        # 3. Update weights
        self.w = self.w + K * error
        
        # 4. Update inverse covariance matrix P
        self.P = (self.P - np.outer(K, np.dot(x, self.P))) / self.lambda_factor
