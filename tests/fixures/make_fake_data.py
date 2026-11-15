# tests/fixtures/make_fake_data.py
import numpy as np
from scipy.sparse import csr_matrix
import json

# 6 users, 5 items. 1 = user liked the item.
# Hand-built so we know the exact Jaccard values.
#
#           item0 item1 item2 item3 item4
#  user0      1     1     0     0     0
#  user1      1     0     1     0     0
#  user2      1     1     1     0     0
#  user3      0     0     1     1     0
#  user4      0     0     0     1     1
#  user5      1     0     0     0     1

data = np.array([
    [1,1,0,0,0],
    [1,0,1,0,0],
    [1,1,1,0,0],
    [0,0,1,1,0],
    [0,0,0,1,1],
    [1,0,0,0,1],
], dtype=np.int8)

mat = csr_matrix(data)

# Save
np.savez("tests/fixtures/fake_matrix.npz",
         by_user=mat,
         by_item=mat.T.tocsr())

# Hand-computed Jaccard values for you to test against
# Set of users for each item:
#   item0 = {0,1,2,5}   |item0| = 4
#   item1 = {0,2}       |item1| = 2
#   item2 = {1,2,3}     |item2| = 3
#   item3 = {3,4}       |item3| = 2
#   item4 = {4,5}       |item4| = 2

expected = {
    (0,1): 1/5,    # {0,2} / {0,1,2,5} = 2/5 = 0.4 ... wait
    # Actually: item0 ∩ item1 = {0,2} = 2, union = {0,1,2,5} = 4
    # Jaccard(0,1) = 2/4 = 0.5
}

# Let me redo this carefully:
# item0 = {0,1,2,5}, item1 = {0,2}, intersection = {0,2} = 2,
# union = {0,1,2,5} = 4, Jaccard = 2/4 = 0.5

expected = {
    (0, 1): 0.5,      # 2/4
    (0, 2): 3/4,      # {0,1,2} / {0,1,2,3,5} = 3/5 = 0.6
    (0, 3): 0.0,      # disjoint
    (0, 4): 0.2,      # {5} / {0,1,2,4,5} = 1/5 = 0.2
    (1, 2): 2/3,      # {0,2} / {0,1,2,3} = 2/4 = 0.5
    (1, 3): 0.0,
    (1, 4): 0.0,
    (2, 3): 2/4,      # {2,3} / {1,2,3} = 2/3
    (2, 4): 0.0,
    (3, 4): 1/3,      # {4} / {3,4,5} = 1/3
}

with open("tests/fixtures/fake_jaccard.json", "w") as f:
    json.dump({f"{a},{b}": v for (a,b), v in expected.items()}, f, indent=2)

print("Done. Check fake_jaccard.json to verify my arithmetic.")