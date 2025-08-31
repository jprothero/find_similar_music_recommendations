from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.sparse.linalg import LinearOperator, cg
import numpy as np
from scipy.optimize import minimize_scalar, minimize

def calculate_ease_for_item_cg(X: csr_matrix, item_id: int, lambda_: float) -> np.ndarray:
    """
    Calculates EASE scores for a single item efficiently without forming the
    full Gram matrix.
    """
    n_users, n_items = X.shape

    # 1. OPTIMIZATION: Calculate the target column `b` directly.
    # This is a sparse matrix-vector product, which is very fast.
    # The result is a dense vector.
    target_col = (X.T @ X[:, item_id]).toarray().flatten()

    # Enforce the diagonal constraint in the target
    # We must add the regularization term here to match the left-hand side
    target_col[item_id] = 0

    # 2. OPTIMIZATION: Define the matrix-vector product for the solver.
    # This function calculates (X.T @ X + λI) @ v without building the matrix.
    def matvec_operator(v):
        # This is calculated as (X.T @ (X @ v)) + (λ * v)
        return X.T @ (X @ v) + lambda_ * v

    # Create the LinearOperator object that the solver can use.
    # It represents the matrix G = (X.T @ X + λI)
    A = LinearOperator(shape=(n_items, n_items), matvec=matvec_operator)

    # 3. Solve the linear system A * x = b using the operator.
    # The solver 'A' is now just a function, not a giant matrix.
    similarities, _ = cg(A, target_col, maxiter=20)
    
    # 4. Enforce the B_jj = 0 constraint in the final result
    similarities[item_id] = 0

    return similarities

def calculate_ease_for_item(X: csr_matrix, item_id: int, lambda_: float, positive=False, fit_intercept=False) -> np.ndarray:
    """
    Calculates the EASE similarity scores for a single item.

    This function solves a Ridge regression problem to find the similarity
    of a target item to all other items.

    Args:
        X (csr_matrix): The user-item interaction matrix of shape (n_users, n_items).
        item_id (int): The column index of the target item.
        lambda_ (float): The L2 regularization parameter. A larger value
                         means stronger regularization.

    Returns:
        np.ndarray: A dense 1D array of shape (n_items,) containing the
                    similarity scores. The score at the target item's own
                    index will be 0.
    """
    # For efficient column slicing, it's better to work with a CSC matrix
    X_csc = X.tocsc()
    
    # Get the target item's column (our y variable)
    y_sparse = X_csc[:, item_id]
    
    # If no user has interacted with this item, all similarities are zero.
    if y_sparse.nnz == 0:
        return np.zeros(X.shape[1])

    # The target vector 'y' must be a dense array for scikit-learn's Ridge model.
    # We use .toarray() to convert it and .flatten() to make it a 1D vector.
    y_dense = y_sparse.toarray().flatten()

    # To enforce the B_jj = 0 constraint in EASE, we remove the
    # target item's column from the training data (our X variable).
    n_items = X.shape[1]
    other_items_mask = np.arange(n_items) != item_id
    # The feature matrix X_train is kept sparse for efficiency
    X_train = X_csc[:, other_items_mask]

    # Initialize and fit the Ridge regression model.
    model = Ridge(alpha=lambda_, fit_intercept=fit_intercept, solver='auto', positive=positive)
    
    # Pass the sparse X and the dense y to the model
    model.fit(X_train, y_dense)

    # The model coefficients are the similarity scores for the *other* items.
    # We need to put them back into a full-sized array.
    similarities = np.zeros(n_items)
    similarities[other_items_mask] = model.coef_

    return similarities

def calculate_ease_for_item_logistic(X: csr_matrix, item_id: int, lambda_: float, fit_intercept=True, use_l1=False) -> np.ndarray:
    """
    Calculates the EASE similarity scores for a single item.

    This function solves a Ridge regression problem to find the similarity
    of a target item to all other items.

    Args:
        X (csr_matrix): The user-item interaction matrix of shape (n_users, n_items).
        item_id (int): The column index of the target item.
        lambda_ (float): The L2 regularization parameter. A larger value
                         means stronger regularization.

    Returns:
        np.ndarray: A dense 1D array of shape (n_items,) containing the
                    similarity scores. The score at the target item's own
                    index will be 0.
    """
    # For efficient column slicing, it's better to work with a CSC matrix
    X_csc = X.tocsc()
    
    # Get the target item's column (our y variable)
    y_sparse = X_csc[:, item_id]
    
    # If no user has interacted with this item, all similarities are zero.
    if y_sparse.nnz == 0:
        return np.zeros(X.shape[1])

    # The target vector 'y' must be a dense array for scikit-learn's Ridge model.
    # We use .toarray() to convert it and .flatten() to make it a 1D vector.
    y_dense = y_sparse.toarray().flatten()

    # To enforce the B_jj = 0 constraint in EASE, we remove the
    # target item's column from the training data (our X variable).
    n_items = X.shape[1]
    other_items_mask = np.arange(n_items) != item_id
    # The feature matrix X_train is kept sparse for efficiency
    X_train = X_csc[:, other_items_mask]
    
    # Initialize and fit the Ridge regression model.
    if use_l1:
        X_sparse = X_train.tocsr()
        X_sparse_32 = csr_matrix((X_sparse.data,
                                  X_sparse.indices.astype(np.int32),
                                  X_sparse.indptr.astype(np.int32)),
                                 shape=X_sparse.shape)
        
        X_train_to_use = X_sparse_32
        
        model = LogisticRegression(C=np.inf if lambda_ == 0 else 1/lambda_, fit_intercept=fit_intercept
                                   , penalty="l1", solver='liblinear'
#                                    , penalty="elasticnet", solver='liblinear'
                                  )
    else:
        X_train_to_use = X_train
        
        model = LogisticRegression(C=np.inf if lambda_ == 0 else 1/lambda_, fit_intercept=fit_intercept)
    
    # Pass the sparse X and the dense y to the model
    model.fit(X_train_to_use, y_dense)

    # The model coefficients are the similarity scores for the *other* items.
    # We need to put them back into a full-sized array.
    similarities = np.zeros(n_items)
    similarities[other_items_mask] = model.coef_[0]

    return similarities

def a_to_b_error_metric(mat, a, b, lambda_, verbose=True):
    similarity_scores_a = calculate_ease_for_item_cg(mat, a, lambda_)
    ranking_a = np.argsort(-similarity_scores_a)
    
    similarity_scores_b = calculate_ease_for_item_cg(mat, b, lambda_)
    ranking_b = np.argsort(-similarity_scores_b)
    
    # lower is better
    error = (ranking_a.tolist().index(b) + ranking_b.tolist().index(a))/2
    
    if verbose:
        print("lambda_:", lambda_)
        print("error:", error)
    
    return error

def optimize_lambda_using_a_to_b_matching(mat, a, b, fast_approximation=True):
    lambda_ = 1000000
    
    prev_error = np.inf
    while True:
        error = a_to_b_error_metric(mat, a, b, lambda_)
        
        if error >= prev_error:
            break
        else:
            lambda_ /= 10
            prev_error = error
            
    if fast_approximation:
        return round(lambda_*10)
    else:
        res = minimize_scalar(lambda lambda_: a_to_b_error_metric(mat, a, b, lambda_), 
                              bracket=(lambda_, lambda_*10, lambda_*100))

        return round(res.x)