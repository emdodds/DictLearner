import numpy as np
import matplotlib.pyplot as plt

"""
Extensions of DictLearner that keep track of how well
they have recovered a known sparse model. The data passed in should
be a StimSet.ToySparseSet object.
"""


def make_fit_learner_class(Learner):
    """Given a particular DictLearner class, returns a version of it that
    keeps track of how well it has recovered a known sparse model."""
    class FitLearner(Learner):
        def initialize_stats(self):
            self.modfits = np.array([])
            Learner.initialize_stats(self)

        def store_statistics(self, *args, **kwargs):
            self.modfits = np.append(self.modfits, self.stims.test_fit(self.Q))
            return Learner.store_statistics(self, *args, **kwargs)

        def get_histories(self):
            histories = Learner.get_histories(self)
            histories['modfits'] = self.modfits
            return histories

        def set_histories(self, histories):
            try:
                self.modfits = histories['modfits']
            except KeyError:
                print('Model fit history not available.')
            Learner.set_histories(self, histories)

        def fit_progress_plot(self, window_size=100, norm=1, start=0, end=-1, 
                              ax=None):
            """Plots a moving average of the error and activity history
            with the given averaging window."""
            window = np.ones(int(window_size))/float(window_size)
            try:
                errorhist = self.errorhist
            except:
                errorhist = self.mse_history
            smoothederror = np.convolve(errorhist[start:end], window, 'valid')
            if norm == 2:
                acthist = self.L2hist
            elif norm == 0:
                acthist = self.L0hist
            else:
                try:
                    acthist = self.L1hist
                except:
                    acthist = self.L1_history
            smoothedactivity = np.convolve(acthist[start:end], window, 'valid')
            smoothedmodfits = np.convolve(self.modfits[start:end], window, 'valid')
            lines = []
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            lines = []
            lines = lines + ax.plot(smoothederror, 'b')
            lines = lines + ax.plot(smoothedactivity, 'g')
            lines = lines + ax.plot(smoothedmodfits, 'r')
            labels = ['MSE', 'L1 activity', 'Model recovery']
            try:
                lam = self.lam
                loss = smoothederror + lam*smoothedactivity
                lines =  lines + ax.plot(loss, 'm')
                labels.append('Sparse coding loss')
            except:
                pass
            ax.legend(lines, labels)
            return ax

    return FitLearner
