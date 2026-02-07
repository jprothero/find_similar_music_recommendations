from scipy.special import expit, logit
from scipy.sparse import csr_array
import numpy as np
from scipy.optimize import minimize_scalar
from ipdb import set_trace

def local_temp_scaling(p, temp):
    p_logit = logit(p)
    
    scaled_logit = p_logit / temp
    
    return expit(scaled_logit)

def npmi_batch(mat, i, temp=1, eps=1e-14, derank_i=True):
    mat = csr_array(mat)
    
    py = mat.mean(axis=0)
    pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps
    
    if temp != 1:
        pxy = local_temp_scaling(pxy, temp)
        py = local_temp_scaling(py, temp)
    
    px = py[i]
    
    npmi = (np.log2(pxy) - (np.log2(px) + np.log2(py))) / -np.log2(pxy)
    
    if derank_i:
        npmi[i] = -np.inf
    
#     npmi[~np.isfinite(npmi)] = -np.inf
    
    return npmi

def make_npmi_batch_popularity_weighted_cache(mat, i, eps=1e-14):
    mat = csr_array(mat)

    py = mat.mean(axis=0)
    pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps

    px = py[i]

    return pxy, py

# def npmi_batch_popularity_weighted(mat, i, alpha, cache=None, eps=1e-14, derank_i=True, normalize=False):
#     if cache is None:
#         mat = csr_array(mat)

#         py = mat.mean(axis=0)
#         pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps
#     else:
#         pxy, py = cache
        
#     px = py[i]

#     pmi = np.log2(pxy) - (np.log2(px) + np.log2(py))

#     if normalize:
#         pmi = pmi / -np.log2(pxy)

#     if derank_i:
#         pmi[i] = -np.inf
        
#     pxy_scaled = pxy**alpha
        
#     return (pmi * pxy_scaled) / pxy_scaled.max()

# def npmi_batch_popularity_weighted(mat, i, alpha, cache=None, eps=1e-14, derank_i=True, normalize=False):
#     alpha = max(alpha, 0)
    
#     if cache is None:
#         mat = csr_array(mat)

#         py = mat.mean(axis=0)
#         pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps

#         px = py[i]

#         pmi = np.log2(pxy) - (np.log2(px) + np.log2(py))
        
#         if normalize:
#             pmi = pmi / -np.log2(pxy)

#         if derank_i:
#             pmi[i] = -np.inf
#     else:
#         pmi, pxy, py = cache
        
#     n_user, n_item = mat.shape

#     rows = n_item
#     cols = n_item
#     N_raw = n_user

#     N_smoothed = N_raw + (alpha * rows * cols)

#     py = py * n_user
#     py = (py + (alpha * cols)) / N_smoothed

#     assert py.min() >= 0
#     assert py.max() <= 1

#     px = py[i]

#     pxy = pxy * n_user
#     pxy = (pxy + alpha)/N_smoothed
    
#     assert pxy.min() >= 0
#     assert pxy.max() <= 1

#     pmi = np.log2(pxy) - (np.log2(px) + np.log2(py))

#     if normalize:
#         pmi = pmi / -np.log2(pxy)

#     if derank_i:
#         pmi[i] = -np.inf
    
#     return pmi

def npmi_batch_popularity_weighted(mat, i, alpha, cache=None, eps=1e-14, derank_i=True, normalize=False):
    alpha = max(alpha, 0)
    
    if cache is None:
        mat = csr_array(mat)

        py = mat.mean(axis=0)
        pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps

        px = py[i]

        pmi = np.log2(pxy) - (np.log2(px) + np.log2(py))
        
        if normalize:
            pmi = pmi / -np.log2(pxy)

        if derank_i:
            pmi[i] = -np.inf
    else:
        pxy, py = cache
        
    n_user, n_item = mat.shape

    # unnormalize back to counts
    py = py * n_user
    
    # add n_user * alpha
    py = py + n_user * alpha
    
    # renormalize to make it a probability
    py = py / (n_user + n_user * alpha)

    assert py.min() >= 0
    assert py.max() <= 1

    px = py[i]
    
    # unnormalize back to counts
    pxy = pxy * n_user
    
    # add alpha
    pxy = pxy + alpha
    
    # renormalize to make it a probability
    pxy = pxy / (n_user + n_user * alpha)
    
    assert pxy.min() >= 0
    assert pxy.max() <= 1

    pmi = np.log2(pxy) - (np.log2(px) + np.log2(py))

    if normalize:
        pmi = pmi / -np.log2(pxy)

    if derank_i:
        pmi[i] = -np.inf
        
    pxy_scaled = pxy**(.001)
        
    return (pmi * pxy_scaled) / pxy_scaled.max()

def a_to_b_error_metric_npmi(mat, a, b, temp, verbose=True):
    similarity_scores_a = npmi_batch(mat, a, temp)
    ranking_a = np.argsort(-similarity_scores_a)
    
    similarity_scores_b = npmi_batch(mat, b, temp)
    ranking_b = np.argsort(-similarity_scores_b)
    
    # lower is better
    error = (ranking_a.tolist().index(b) + ranking_b.tolist().index(a))/2
    
    if verbose:
        print("temp:", temp)
        print("error:", error)
    
    return error

def a_to_b_error_metric_npmi_pop_weighted(mat, a, b, alpha, alpha_penalty=False, cache_a=None, cache_b=None, verbose=True):
    similarity_scores_a = npmi_batch_popularity_weighted(mat, a, alpha, cache=cache_a)
    ranking_a = np.argsort(-similarity_scores_a)
    
    similarity_scores_b = npmi_batch_popularity_weighted(mat, b, alpha, cache=cache_b)
    ranking_b = np.argsort(-similarity_scores_b)
    
    # lower is better
    if alpha_penalty:
        error = (ranking_a.tolist().index(b) + ranking_b.tolist().index(a))/2 + min(.499999, max(alpha, 0))
    else:
        error = (ranking_a.tolist().index(b) + ranking_b.tolist().index(a))/2
    
    if verbose:
        print("alpha:", alpha)
        print("error:", error)
    
    return error

def optimize_alpha_using_a_to_b_matching_npmi(mat, a, b, alpha_penalty=True, verbose=True):
    def f(alpha, 
          cache_a=make_npmi_batch_popularity_weighted_cache(mat, a),
          cache_b=make_npmi_batch_popularity_weighted_cache(mat, b)
         ):
        
        return a_to_b_error_metric_npmi_pop_weighted(mat, a, b, alpha, 
                                                     cache_a=cache_a, cache_b=cache_b, alpha_penalty=alpha_penalty)
    
    return minimize_scalar(f).x