"""LSH banding on MinHash signatures (Member 3).

A signature of k = b * r values is cut into b bands of r rows. Two items become candidates if they
agree on ALL r rows of at least one band. For items with Jaccard similarity s:

    P(candidate) = 1 - (1 - s**r)**b          (the S-curve)

and the curve is steepest near the threshold (1/b)**(1/r).

Buckets are exact: a band's r values are compared as a whole (no extra hashing), so two items share a
bucket only if their r signature values are identical.
"""
import numpy as np

SETTINGS = [(64, 2), (32, 4), (16, 8), (8, 16)]  # (bands, rows); every one uses k = b * r = 128


def theoretical_probability(s, b, r):
    """S-curve: chance that a pair with Jaccard similarity s lands in a common bucket."""
    s = np.asarray(s, dtype=float)
    return 1.0 - (1.0 - s**r) ** b


def threshold(b, r):
    """Similarity at which the S-curve rises fastest."""
    return (1.0 / b) ** (1.0 / r)


class LSHIndex:
    """Banded buckets for one (b, r) setting.

    Per band we keep: bucket_of[band, item] (bucket id, -1 for items without users), plus the items
    sorted by bucket (order[band]) and bucket start offsets (starts[band]) so a bucket's members are a slice.
    """

    def __init__(self, sig, b, r):
        n_items, k = sig.shape
        if b * r != k:
            raise ValueError(f'b*r must equal the signature length ({b}*{r} != {k})')
        self.b, self.r, self.n_items = b, r, n_items
        valid = np.any(sig != np.uint32(0xFFFFFFFF), axis=1)  # items with at least one user
        self.valid = valid
        valid_idx = np.flatnonzero(valid)
        self.bucket_of = np.full((b, n_items), -1, dtype=np.int32)
        self.order, self.starts = [], []
        for band in range(b):
            block = np.ascontiguousarray(sig[valid_idx, band * r:(band + 1) * r])
            as_rows = block.view(np.dtype((np.void, block.dtype.itemsize * r))).ravel()
            _, inverse = np.unique(as_rows, return_inverse=True)  # identical r-tuples share an id
            inverse = inverse.ravel().astype(np.int32)
            self.bucket_of[band, valid_idx] = inverse
            srt = np.argsort(inverse, kind='stable')
            counts = np.bincount(inverse)
            self.order.append(valid_idx[srt].astype(np.int32))
            self.starts.append(np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))

    def candidates(self, item):
        """Sorted array of items sharing at least one band bucket with `item` (the item itself excluded)."""
        mark = np.zeros(self.n_items, dtype=bool)
        for band in range(self.b):
            bucket = self.bucket_of[band, item]
            if bucket < 0:
                continue
            s = self.starts[band]
            mark[self.order[band][s[bucket]:s[bucket + 1]]] = True
        mark[item] = False
        return np.flatnonzero(mark).astype(np.int32)

    def is_candidate(self, i, j):
        """For index arrays i, j: True where the pair shares a bucket in at least one band."""
        bi, bj = self.bucket_of[:, i], self.bucket_of[:, j]
        return np.any((bi == bj) & (bi >= 0), axis=0)

    def nbytes(self):
        return int(self.bucket_of.nbytes + sum(o.nbytes for o in self.order) + sum(s.nbytes for s in self.starts))

    def save(self, path):
        arrays = {'bucket_of': self.bucket_of, 'b': self.b, 'r': self.r}
        for band in range(self.b):
            arrays[f'order_{band}'], arrays[f'starts_{band}'] = self.order[band], self.starts[band]
        np.savez_compressed(path, **arrays)
