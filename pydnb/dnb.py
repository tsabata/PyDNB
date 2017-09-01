import time

import numpy as np
import pandas as pd


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

    def mle(self, df, state_col, features=None, avoid_zeros=False, fix_scales=False):
        t = time.process_time()
        """ Fitting dynamics in the DNB """
        self._dynamic_mle(df[state_col], avoid_zeros)
        """ Fitting observable variables """
        self.B = {}
        for st in self.states_list:
            self._features_mle(df[df[state_col] == st].drop([state_col], axis=1), st, features)
        if fix_scales:
            self.fix_zero_scale()
        if self.debug:
            elapsed_time = time.process_time() - t
            print("MLE finished in %d seconds." % elapsed_time)
        return self

    def _dynamic_mle(self, df, avoid_zeros):
        states_vec = df.as_matrix()
        self.states_list = np.unique(states_vec)
        states_nr = len(self.states_list)
        self.A = np.zeros((states_nr, states_nr))
        if avoid_zeros:
            self.A += 1
        self.states_prior = np.zeros(states_nr)
        self.states_prior[self._state_index(states_vec[0])] += 1
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

    def fix_zero_scale(self, new_scale=1, tolerance=0.000001):
        for state in self.states_list:
            for f, dist in self.features.items():
                scale = self.B[(state, f)][-1]
                if scale < tolerance:
                    if self.debug:
                        print("state: %s,feature: %s" % (str(state), str(f)))
                    self.B[(state, f)][-1] = new_scale

    def prior_prob(self, state, log=False):
        if log:
            return np.log(self.states_prior[self._state_index(state)])
        else:
            return self.states_prior[self._state_index(state)]

    def emission_prob(self, state, data, log=False):
        prob = 1
        if log:
            prob = np.log(prob)
        for f, dist in self.features.items():
            arg = self.B[(state, f)][:-2]
            loc = self.B[(state, f)][-2]
            scale = self.B[(state, f)][-1]
            if log:
                prob += dist.logpdf(data[f], loc=loc, scale=scale, *arg)
            else:
                prob *= dist.pdf(data[f], loc=loc, scale=scale, *arg)
        return prob

    def transition_prob(self, state1, state2, log=False):
        if log:
            return np.log(self.A[self._state_index(state1), self._state_index(state2)])
        else:
            return self.A[self._state_index(state1), self._state_index(state2)]

    def _forward(self, data, k=None, state=None):
        alpha = np.zeros((len(self.states_list), len(data)))
        """ alpha t=0 """
        for st in self.states_list:
            alpha[self._state_index(st)] = self.prior_prob(st, log=True) + self.emission_prob(st, data.iloc[0],
                                                                                              log=True)
        for t in range(1, len(data)):
            for st in self.states_list:
                alpha[self._state_index(st)][t] = sum(
                    alpha[self._state_index(_st)][t - 1] + self.transition_prob(_st, st, log=True) for _st in
                    self.states_list) + self.emission_prob(st, data.iloc[t], log=True)
        if state:
            alpha = alpha[self._state_index(state), :]
        if k:
            alpha = alpha[:, k]
        return alpha

    def _backward(self, data, k=None, state=None):
        beta = np.zeros((len(self.states_list), len(data)))
        for t in range(len(data) - 1, 0, -1):
            for st in self.states_list:
                beta[self._state_index(st)][t] = sum(
                    self.transition_prob(st, _st, log=True) + self.emission_prob(_st, data.iloc[t + 1], log=True) +
                    beta[_st][t + 1] for _st
                    in self.states_list)
        if state:
            beta = beta[self._state_index(state), :]
        if k:
            beta = beta[:, k]
        return beta

    def sample(self, size, n=1):
        sequences = []
        for i in range(n):
            Y, output = [], {}
            state = self.states_list[np.random.choice(len(self.states_list), 1, p=self.states_prior)[0]]
            for _ in range(size):
                for f, dist in self.features.items():
                    arr = output.get(f, [])
                    arg = self.B[(state, f)][:-2]
                    loc = self.B[(state, f)][-2]
                    scale = self.B[(state, f)][-1]
                    arr.append(dist(loc=loc, scale=scale, *arg).rvs())
                    output[f] = arr
                Y.append(state)
                state = self.states_list[
                    np.random.choice(len(self.states_list), 1, p=self.A[self._state_index(state)])[0]]
            df = pd.DataFrame({**{'state': Y}, **output})
            sequences.append(df)
        return sequences

    def obs_seq_probability(self, data):
        return sum(self._forward(data, k=len(data) - 1))

    def seq_probability(self, data, path, log=True):
        prob = 0
        path = list(path)
        prob += self.prior_prob(path[0], log=True)
        prob += self.emission_prob(path[0], data.iloc[0], log=True)
        for t in range(1, len(data)):
            prob += self.transition_prob(path[t - 1], path[t], log=True)
            prob += self.emission_prob(path[t], data.iloc[t], log=True)
        if not log:
            return np.exp(prob)
        return prob

    def viterbi(self, data):
        pass
