import time

import numpy as np


class DNB:
    """
    Class representing Dynamic Naive Bayes.
    """

    def __init__(self, debug=False):
        self.state_prior = None
        self.states_list = None
        self.A = None
        self.B = None
        self.debug = debug

    def mle(self, df, state_col, features=None):
        t = time.process_time()
        self._dynamic_mle(df[state_col])
        for st in self.states_list:
            self._features_mle(df, st)
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
            self.A[np.searchsorted(self.states_list, states_vec[i - 1]), np.searchsorted(self.states_list,
                                                                                         states_vec[i])] += 1
            self.states_prior[np.searchsorted(self.states_list, states_vec[i])] += 1
        self.states_prior = self.states_prior / self.states_prior.sum()
        for i in range(states_nr):
            self.A[i] = self.A[i] / self.A[i].sum()

    def _features_mle(self, df, state):
        pass
