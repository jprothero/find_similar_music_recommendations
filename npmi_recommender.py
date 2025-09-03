from scipy.special import expit, logit
from scipy.sparse import csr_array
import numpy as np
from scipy.optimize import minimize_scalar

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

def make_npmi_batch_popularity_weighted_cache(mat, i, eps=1e-14, derank_i=True):
    mat = csr_array(mat)

    py = mat.mean(axis=0)
    pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps

    px = py[i]

    npmi = (np.log2(pxy) - (np.log2(px) + np.log2(py))) / -np.log2(pxy)

    if derank_i:
        npmi[i] = -np.inf

    return npmi, pxy

def npmi_batch_popularity_weighted(mat, i, lambda_, cache=None, eps=1e-14, derank_i=True):
    if cache is None:
        mat = csr_array(mat)

        py = mat.mean(axis=0)
        pxy = (mat[:, i:i+1] * mat).mean(axis=0) + eps

        px = py[i]

        npmi = (np.log2(pxy) - (np.log2(px) + np.log2(py))) / -np.log2(pxy)

        if derank_i:
            npmi[i] = -np.inf
    else:
        npmi, pxy = cache
        
    pxy_scaled = pxy**lambda_
        
    return (npmi * pxy_scaled) / pxy_scaled.max()

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

def a_to_b_error_metric_npmi_pop_weighted(mat, a, b, lambda_, lambda_penalty=False, cache_a=None, cache_b=None, verbose=True):
    similarity_scores_a = npmi_batch_popularity_weighted(mat, a, lambda_, cache=cache_a)
    ranking_a = np.argsort(-similarity_scores_a)
    
    similarity_scores_b = npmi_batch_popularity_weighted(mat, b, lambda_, cache=cache_b)
    ranking_b = np.argsort(-similarity_scores_b)
    
    # lower is better
    if lambda_penalty:
        error = ranking_a.tolist().index(b) + ranking_b.tolist().index(a) + min(1, max(lambda_, 0))
    else:
        error = (ranking_a.tolist().index(b) + ranking_b.tolist().index(a))/2
    
    if verbose:
        print("lambda_:", lambda_)
        print("error:", error)
    
    return error

def optimize_lambda_using_a_to_b_matching_npmi(mat, a, b, optimize_rank=True, verbose=True):
    def f(lambda_, 
          cache_a=make_npmi_batch_popularity_weighted_cache(mat, a) if optimize_rank else None,
          cache_b=make_npmi_batch_popularity_weighted_cache(mat, b) if optimize_rank else None
         ):
        
        if optimize_rank:
            return a_to_b_error_metric_npmi_pop_weighted(mat, a, b, lambda_, cache_a=cache_a, cache_b=cache_b, lambda_penalty=True)
        else:
            npmi_a = npmi_batch_popularity_weighted(mat, a, lambda_)
            npmi_b = npmi_batch_popularity_weighted(mat, b, lambda_)

            score = ((npmi_a[b] / npmi_a.max()) + (npmi_b[a] / npmi_b.max()))/2

            if verbose:
                print("score:", score)

            return -score
    
    return minimize_scalar(f).x