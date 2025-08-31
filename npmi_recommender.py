from scipy.special import expit, logit
from scipy.sparse import csr_array
import numpy as np

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