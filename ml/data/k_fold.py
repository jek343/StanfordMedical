

def k_fold(X, y, k):
    out = []
    p = X.shape[0] / k
    for i in range(k):
        train_inds = list(range(0, int(p * i)))
        train_inds.extend(list(range(int(p * i + p), X.shape[0])))
        test_inds = list(range(int(p * i), int(p * i + p)))
        out.append((X[train_inds], y[train_inds], X[test_inds], y[test_inds]))
    return out
