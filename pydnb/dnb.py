import time

import numpy as np


class DNB:
    """
    Class representing Dynamic Naive Bayes.
    """

    def __init__(self, debug=False):
        self.states_prior = None
        self.states_list = None
        self.features_list = None
        self.A = None
        self.B = None
        self.debug = debug

    def mle(self, df, state_col, features=None):
        t = time.process_time()
        """ Fitting dynamics in the DNB """
        self._dynamic_mle(df[state_col])
        """ Fitting observable variables """
        self.B = {}
        for st in self.states_list:
            self._features_mle(df[df[state_col]==st].drop([state_col],axis=1),st)
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
        """simplified_version"""
        import scipy.stats as st
        self.features_list = list(df.columns.values)
        for f in self.features_list:
            params = st.norm.fit(df[f])
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            if self.debug:
                print("%s, %s, %s"%(str(arg),str(loc),str(scale)))
            self.B[(state, f)] = [st.norm] + list(params)

    def emission_prob(self,state,data):
        prob = 1
        for f in self.features_list:
            dist = self.B[(state,f)][0]
            arg = self.B[(state,f)][1:-2]
            loc = self.B[(state,f)][-2]
            scale = self.B[(state,f)][-1]
            prob *= dist.pdf(data[f], loc=loc, scale=scale, *arg)
        return prob

    def transition_prob(self,state1, state2):
        return self.A[np.searchsorted(self.states_list, state1), np.searchsorted(self.states_list, state2)]

    def _forward(self,data,k=None,state=None):
        pass

    def _backward(self,data,k=None,state=None):
        pass

    def viterbi(self, data):
        pass

    def seq_probability(self, data, path):
        pass
