import time

import numpy as np


class DNB:
    """
    Class representing Dynamic Naive Bayes.
    """

    def __init__(self, debug=False):
        self.states_prior = None
        self.states_list = None
        self.features = None
        self.A = None
        self.B = None
        self.debug = debug

    def _state_index(self, state):
        return np.searchsorted(self.states_list, state)

    def mle(self, df, state_col, features=None):
        t = time.process_time()
        """ Fitting dynamics in the DNB """
        self._dynamic_mle(df[state_col])
        """ Fitting observable variables """
        self.B = {}
        for st in self.states_list:
            self._features_mle(df[df[state_col] == st].drop([state_col], axis=1), st, features)
        if self.debug:
            elapsed_time = time.process_time() - t
            print("MLE finished in %d seconds." % elapsed_time)

    def _dynamic_mle(self, df):
        states_vec = df.as_matrix()
        self.states_list = np.unique(states_vec)
        states_nr = len(self.states_list)
        self.A = np.zeros((states_nr, states_nr))
        self.states_prior = np.zeros(states_nr)
        self.states_prior[np.searchsorted(self.states_list, states_vec[0])] += 1
        for i in range(1, len(states_vec)):
            self.A[self._state_index(states_vec[i - 1]), self._state_index(states_vec[i])] += 1
            self.states_prior[self._state_index(states_vec[i])] += 1
        self.states_prior = self.states_prior / self.states_prior.sum()
        for i in range(states_nr):
            self.A[i] = self.A[i] / self.A[i].sum()

    def _features_mle(self, df, state, features):
        import scipy.stats as st
        if features is None:
            self.features = dict.fromkeys(list(df.columns.values), st.norm)
        else:
            self.features = features
        for f, dist in self.features.items():
            params = dist.fit(df[f])
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            if self.debug:
                print("Distribution: %s, args: %s, loc: %s, scale: %s" % (str(dist), str(arg), str(loc), str(scale)))
            self.B[(state, f)] = list(params)

    def prior_prob(self, state):
        return self.states_prior[self._state_index(state)]

    def emission_prob(self, state, data):
        prob = 1
        for f, dist in self.features.items():
            arg = self.B[(state, f)][:-2]
            loc = self.B[(state, f)][-2]
            scale = self.B[(state, f)][-1]
            prob *= dist.pdf(data[f], loc=loc, scale=scale, *arg)
        return prob

    def transition_prob(self, state1, state2):
        return self.A[self._state_index(state1), self._state_index(state2)]

    def _forward(self, data, k=None, state=None):
        alpha = np.zeros((len(self.states_list), len(data)))
        """ alpha t=0 """
        for st in self.states_list:
            alpha[self._state_index(st)] = self.prior_prob(st) * self.emission_prob(st, data.iloc[0])

        for t in range(1, len(data)):
            for st in self.states_list:
                alpha[self._state_index(st)][t] = sum(
                    alpha[self._state_index(_st)][t - 1] * self.transition_prob(_st, st) for _st in
                    self.states_list) * self.emission_prob(st,data.iloc[t])
        if state:
            alpha = alpha[self._state_index(state), :]
        if k:
            alpha = alpha[:, k]
        return alpha

    def _backward(self, data, k=None, state=None):
        pass

    def viterbi(self, data):
        pass

    def seq_probability(self, data, path):
        pass
