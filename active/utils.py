import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import copy

def ActualSequentialLoader(subsetRandomLoader, indices=None, transform=None, batch_size=None):
    indices = indices if indices is not None else subsetRandomLoader.sampler.indices
    train_sampler = ActualSequentialSampler(indices)
    dataset = copy.deepcopy(subsetRandomLoader.dataset)
    if transform is not None:
        dataset.transform = transform

    batch_size = batch_size if batch_size is not None else subsetRandomLoader.batch_size
    actualSequentialLoader = torch.utils.data.DataLoader(dataset, sampler=train_sampler,
                                                         num_workers=subsetRandomLoader.num_workers,
                                                         batch_size=batch_size, drop_last=False)
    return actualSequentialLoader

class ActualSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.
    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

def outer_product_opt(c1, d1, c2, d2):
    """Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
    """
    B1, B2 = c1.shape[0], c2.shape[0]
    t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
    t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
    t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
    t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
    t2 = t2.reshape(1, B2).repeat(B1, axis=0)
    return t1 + t2 - 2*t3

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
    """Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
    Parameters
    ----------
    X1, X2 : array or sparse matrix
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters : integer
        The number of seeds to choose
    init : list
        List of points already picked
    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """

    n_samples, n_feat1 = X1.shape
    _, n_feat2 = X2.shape
    # x_squared_norms = row_norms(X, squared=True)
    centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
    centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

    idxs = np.empty((n_clusters+len(init)-1,), dtype=np.long)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = init

    centers1[:len(init)] = X1[center_id]
    centers2[:len(init)] = X2[center_id]
    idxs[:len(init)] = center_id

    # Initialize list of closest distances and calculate current potential
    distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

    candidates_pot = distance_to_candidates.sum(axis=1)
    best_candidate = np.argmin(candidates_pot)
    current_pot = candidates_pot[best_candidate]
    closest_dist_sq = distance_to_candidates[best_candidate]

    # Pick the remaining n_clusters-1 points
    for c in range(len(init), len(init)+n_clusters-1):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        idxs[c] = best_candidate

    return None, idxs[len(init)-1:]

def get_embedding(model, loader, device, num_classes, cfg, with_emb=False, emb_dim=512):
    model.eval()
    embedding = torch.zeros([len(loader.sampler), num_classes])
    embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
    labels = torch.zeros(len(loader.sampler))
    preds = torch.zeros(len(loader.sampler))
    batch_sz = cfg.DATALOADER.BATCH_SIZE
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if with_emb:
                e1, e2 = model(data, with_emb=True)
                embedding_pen[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
            else:
                e1 = model(data, with_emb=False)

            embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
            labels[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
            preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

    return embedding, labels, preds, embedding_pen
