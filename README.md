# DictLearner
A base class for and examples of algorithms that learn a dictionary to represent data.

Currently there is a base class, which implements gradient descent on mean-squared reconstruction error, and one example which uses a locally competitive algorithm (LCA) for inference and the base class's "learn" method. My SAILnet implementation (see http://github.com/emdodds/SAILnet also extends DictLearner.

StimSet provides containers for the sets of stimuli that a DictLearner learns from, including functionality to create pictures of these stimuli or of dictionary elements.
