import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_hidden_states(path):
    """
    Load hidden states from a saved file.
    Returns:
        hidden_dict: dict with keys such as 'patch_tokens', 'cls_tokens', 'hidden_states', 'labels'
    """
    hidden_dict = torch.load(path, map_location='cpu')
    return hidden_dict

# ------------------------------------------------------------------------
#  1) Strictly Aligned Unbiased & Biased HSIC as in the Original Paper
# ------------------------------------------------------------------------

def hsic_unbiased_strict(K, L):
    """
    Unbiased HSIC estimator strictly following Song et al. (2012, Equation 5).
    Note:
      - The diagonal elements K_{ii} and L_{ii} should be set to 0 externally (because we sum over i != j).
      - Although this formula operates on the entire matrix, in CKNNA we pass a submatrix (with a mask after neighbor intersection),
        which is equivalent to a local HSIC.
    """
    n = K.shape[0]
    if L.shape[0] != n or L.shape[1] != n:
        raise ValueError("K, L must have the same shape for HSIC computation.")

    # Compute sum_{i != j} K_{ij} * L_{ij}
    sumKL = torch.sum(K * L)

    # Compute sum_{i != j} K_{ij} and sum_{i != j} L_{ij}
    sumK = torch.sum(K)
    sumL = torch.sum(L)

    # Below corresponds to the last term in the original formula: 2/(n-2) * sum(~K_{ij} * ~L_{ij})
    # For simplicity, many implementations perform the operation on ~K (i.e., "K minus its row/column mean plus the overall mean")
    # in a single tensor operation. However, many reference codes use an operation like (K @ L) instead.
    # The following code is equivalent to:
    #   - 2/(n-2) * \sum_{i,j} K'_{ij} L'_{ij}, where K' = \tilde{K}
    # Here we directly reuse K and L: first compute the dot product:
    #   torch.sum(K @ L) = \sum_{i,k} K_{i,k} L_{k,i} (note that L must be transposed)
    # However, since Song12 eqn(5) is written as \sum_{i \neq j} K_{ij} L_{ij}, one must carefully compare indices i,k versus i,j.
    # To align with many popular implementations: torch.mm(K, L) yields an [n, n] matrix M where M_{i,j} = \sum_k K_{i,k} * L_{k,j}.
    # To ensure M_{i,i} = \sum_{k} K_{i,k}*L_{k,i}, and finally sum(M) = \sum_{i,j} M_{i,j},
    # this is an equivalent reformulation of the common term \sum_{i,j} \tilde{K}_{ij}\tilde{L}_{ij}.
    # However, note that eqn(5) uses L_{ij}, while torch.mm(K, L) computes the product of K_{i,*} and L_{*,j}.
    # Therefore, to align indices (i, j), L is transposed: L -> L.T
    # Thus, M_{i,j} = \sum_k K_{i,k} L_{j,k}, aligning the indices, and sum(M) = \sum_{i,j}\sum_k K_{i,k} L_{j,k}.
    # Below is the approach used in many open-source implementations:
    M = torch.mm(K, L.T)  # [n, n]
    sumKmmL = torch.sum(M)

    # Equation (5):
    # HSIC = [ sum_{i!=j} K_{ij}L_{ij} + (sum_{i!=j} K_{ij})(sum_{i!=j} L_{ij}) / ((n-1)(n-2)) - 2/(n-2)* sum_{i!=j} \tilde{K}_{ij}\tilde{L}_{ij} ]
    #        / [ n (n-3) ]
    # The following is a vectorized implementation based on existing code.
    # Note: In the CKNNA scenario, K and L may already have the diagonal set to 0 and non-neighbor entries set to 0.
    # Thus, when i=j or for entries not in the neighbor intersection, they are 0 and do not contribute to the sum.
    HSIC_val = (
        sumKL
        + (sumK * sumL) / ((n - 1) * (n - 2))
        - (2.0 / (n - 2)) * sumKmmL
    ) / (n * (n - 3))

    return HSIC_val

def hsic_biased_strict(K, L):
    """
    Strict biased HSIC estimator: 1/(n^2) * Tr(KHLH),
    where H = I - 1/n * (1 1^T).
    This is the common form used in traditional CKA (the coefficient might be 1/n^2 or 1/(n-1)^2).
    """
    n = K.shape[0]
    if L.shape[0] != n or L.shape[1] != n:
        raise ValueError("K, L must have the same shape for HSIC computation.")

    H = torch.eye(n, dtype=K.dtype, device=K.device) - (1.0 / n)
    KH = K @ H
    LH = L @ H
    HSIC_val = torch.trace(KH @ LH) / (n ** 2)
    return HSIC_val

# ------------------------------------------------------------------------
# 2) Core Function for CKNNA Based on "Neighbor Intersection":
#    Using inner product kernel + top-k neighbor intersection + HSIC normalization
# ------------------------------------------------------------------------
def compute_cknna(vec1, vec2, k=10, unbiased=True):
    """
    Compute the CKNNA score (neighborhood-level aligned CKA), strictly using the inner product kernel and Song12 version of HSIC.
    vec1, vec2: numpy arrays with shape [N, D]

    Steps:
     1) Convert to tensors: feats_A, feats_B = torch.from_numpy(vec1) and torch.from_numpy(vec2)
     2) Compute kernels: K = feats_A @ feats_A^T, L = feats_B @ feats_B^T
     3) For each sample i, find the top-k neighbors in K and L (excluding self)
     4) Construct a neighbor mask: set positions to 1 if they are mutual neighbors in both K and L, 0 otherwise
     5) Compute maskedK = K * mask and maskedL = L * mask (set the diagonal to 0 since i != j)
     6) Compute the HSIC (unbiased or biased) on maskedK and maskedL
     7) Similarly compute val_kk = HSIC(maskedK, maskedK) and val_ll = HSIC(maskedL, maskedL)
     8) Normalize in CKA style: cknna = val_kl / sqrt(val_kk * val_ll + 1e-10)

    Returns: cknna_score (float)
    """
    feats_A = torch.from_numpy(vec1).float()
    feats_B = torch.from_numpy(vec2).float()
    feats_A = torch.nn.functional.normalize(feats_A, p=2, dim=1) 
    feats_B = torch.nn.functional.normalize(feats_B, p=2, dim=1)
    device = feats_A.device
    n = feats_A.shape[0]

    if k < 1 or k >= n:
        raise ValueError("k should be in the range [1, n-1]")

    # 1) Compute the inner product kernel
    K = feats_A @ feats_A.T  # [n, n]
    L = feats_B @ feats_B.T  # [n, n]

    # 2) Set the diagonal to -inf to exclude self as neighbor
    #    (If you wish to include self as a neighbor, modify accordingly)
    K_for_nn = K.clone()
    L_for_nn = L.clone()
    K_for_nn.fill_diagonal_(float('-inf'))
    L_for_nn.fill_diagonal_(float('-inf'))

    # 3) Find the top-k neighbors (in descending order of similarity)
    valsK, idxK = torch.topk(K_for_nn, k, dim=1)
    valsL, idxL = torch.topk(L_for_nn, k, dim=1)

    # 4) Construct the neighbor mask:
    #    maskK[i, j] = 1 indicates that j is among the top-k neighbors of i in K
    #    maskL[i, j] = 1 indicates that j is among the top-k neighbors of i in L
    maskK = torch.zeros(n, n, device=device)
    maskL = torch.zeros(n, n, device=device)
    row_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(idxK)
    maskK.scatter_(1, idxK, 1.0)
    maskL.scatter_(1, idxL, 1.0)

    # 5) Compute the intersection of neighbors: only positions where j is a neighbor in both K and L are 1
    #    (For a bidirectional neighbor requirement, one could further check mask[i,j] & mask[j,i] if needed)
    mask = maskK * maskL

    # 6) Compute maskedK and maskedL, and set the diagonal to 0
    maskedK = K * mask
    maskedL = L * mask
    maskedK.fill_diagonal_(0.0)
    maskedL.fill_diagonal_(0.0)

    # 7) Compute HSIC for kl, kk, and ll
    if unbiased:
        val_kl = hsic_unbiased_strict(maskedK, maskedL)
        val_kk = hsic_unbiased_strict(maskedK, maskedK)
        val_ll = hsic_unbiased_strict(maskedL, maskedL)
    else:
        val_kl = hsic_biased_strict(maskedK, maskedL)
        val_kk = hsic_biased_strict(maskedK, maskedK)
        val_ll = hsic_biased_strict(maskedL, maskedL)

    # 8) Normalize in CKA style
    denom = torch.sqrt(val_kk * val_ll + 1e-10)
    cknna_val = (val_kl / denom).item()

    return cknna_val


def main():
    parser = argparse.ArgumentParser(description="Compare hidden states of two models using strictly-defined CKNNA.")
    parser.add_argument("--hidden1", type=str, required=True,
                        help="Path to the first hidden states file (e.g. SiT_L2R).")
    parser.add_argument("--hidden2", type=str, required=True,
                        help="Path to the second hidden states file (e.g. DINOv2).")
    parser.add_argument("--output-heatmap", type=str, default="cknna_heatmap.pdf",
                        help="Path to save the CKNNA heatmap.")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Number of points to sample from each hidden state for similarity computation.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors for CKNNA.")
    parser.add_argument("--unbiased", action='store_true',
                        help="Use the unbiased HSIC formula (Equation (5) from Song12). Default=False (biased).")

    args = parser.parse_args()

    # 1) Load hidden states
    hidden1 = load_hidden_states(args.hidden1)
    hidden2 = load_hidden_states(args.hidden2)
    
    # Print the structure of hidden states for the first model
    print("Hidden1 Layers and Structures:")
    for key in sorted(hidden1.keys()):
        value = hidden1[key]
        if isinstance(value, dict):
            print(f"{key}: dict with keys {list(value.keys())}")
        else:
            print(f"{key}: {type(value)}")

    print("\nHidden2 Layers and Structures:")
    for key in sorted(hidden2.keys()):
        value = hidden2[key]
        if isinstance(value, dict):
            print(f"{key}: dict with keys {list(value.keys())}")
        else:
            print(f"{key}: {type(value)}")

    # 2) Extract the required hidden_states
    #    - For SiT_L2R, they might be stored in hidden1['hidden_states']
    #    - For DINOv2, it may contain keys such as 'patch_tokens', 'cls_tokens', 'hidden_states'
    if 'hidden_states' in hidden1:
        hidden1_layers = hidden1['hidden_states']
    else:
        raise KeyError("hidden1 does not contain 'hidden_states' key.")

    if 'last_layer_outputs' in hidden2: 
        last_layer_outputs = hidden2['last_layer_outputs']  # Assuming shape [N, D] 
    else: 
        raise KeyError("hidden2 does not contain 'last_layer_outputs' key.")

    hidden1_layer_keys = sorted(hidden1_layers.keys())

    num_sit_layers = len(hidden1_layer_keys)

    print(f"\nSiT_L2R model has {num_sit_layers} layers.")

    cknna_scores = []
    layer_labels = []

    # Compare corresponding layers
    for i, layer_key_l2r in enumerate(hidden1_layer_keys):
        tensor1 = hidden1_layers[layer_key_l2r]  # e.g. [N, 256, D]
        tensor2 = last_layer_outputs  # e.g. [N, 257, D]
        
        # Apply average pooling
        tensor1_pooled = tensor1.mean(dim=1)  # [N, D]
        tensor2_pooled = tensor2.mean(dim=1)  # [N, D]

        # Check sample consistency
        if tensor1_pooled.size(0) != tensor2_pooled.size(0):
            print(f"[Warning] Mismatch in sample numbers, skipping {layer_key_l2r} vs last_layer_outputs")
            continue
        
        # Randomly sample according to batch size
        n_samples = tensor1_pooled.size(0)
        if n_samples > args.batch_size:
            idx = np.random.choice(n_samples, args.batch_size, replace=False)
            tensor1_pooled = tensor1_pooled[idx]
            tensor2_pooled = tensor2_pooled[idx]

        # Convert to numpy arrays
        vec1 = tensor1_pooled.cpu().numpy()
        vec2 = tensor2_pooled.cpu().numpy()

        # Compute CKNNA
        cknna_val = compute_cknna(vec1, vec2, k=args.k, unbiased=args.unbiased)
        cknna_scores.append(cknna_val)
        layer_labels.append(f"{layer_key_l2r} vs last_layer_outputs")

        print(f"CKNNA({layer_key_l2r}, last_layer_outputs): {cknna_val:.4f}")

if __name__ == "__main__":
    main()