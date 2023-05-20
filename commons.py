import numpy as np

def compute_entropy(sampleSplit):
    # If there is only only one class, the entropy is 0
    if len(sampleSplit) < 2:
        return 0
    else:
        freq = np.array(sampleSplit.value_counts(normalize=True))
        return -(freq * np.log2(freq)).sum()

def compute_info_gain(sampleAttribute, sample_target):

    values = sampleAttribute.value_counts(normalize=True)
    split_ent = 0

    # Iterate for each class of the sample attribute
    for v, fr in values.items():

        index = sampleAttribute == v
        sub_ent = compute_entropy(sample_target[index])

        # Weighted sum of the entropies
        split_ent += fr * sub_ent

    # Compute the entropy without any split
    ent = compute_entropy(sample_target)
    return ent - split_ent
