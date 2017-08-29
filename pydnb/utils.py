import numpy as np


def kullback_leibler_distance(dnb1, dnb2):
    raise NotImplementedError('The method kullback_leibler_distance is not implemented.')
    distance = 0
    trans_div = np.log(dnb1.A / dnb2.A)


def output_sequence_distance(dnb1, dnb2, seq_len=100, sequences=10):
    seqs1 = dnb1.sample(seq_len, n=sequences)
    prob1 = sum([dnb2.seq_probability(s, s.state) for s in seqs1]) / sequences / seq_len
    seqs2 = dnb2.sample(seq_len, n=sequences)
    prob2 = sum([dnb1.seq_probability(s, s.state) for s in seqs2]) / sequences / seq_len
    return (prob1 + prob2) / -2
