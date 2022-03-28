import numpy as np
from hmm import ProbabilityVector, ProbabilityMatrix, HiddenMarkovLayer
from get_words import get_all_words

def verify_hmm_complexity(word, size, complexity, proportion=0.5, epsilon=0):
    observables = [str(n) for n in range(size)]

    print('Verifying that the complexity of {} is > {}'. format(''.join(word), complexity))

    # generate a random sample of indices of hmms
    hmms = np.genfromtxt('{}_verification_complexity_{}_size_{}.txt'.format(''.join(word), complexity, size))
    all_start_indices = np.where(hmms[:,0] == -1)[0]
    start_indices = np.random.choice(all_start_indices, int(proportion*len(all_start_indices)), replace=False)

    # get and format the words to test
    words = [curr_word for curr_word in get_all_words(len(word), size) if curr_word != word]
    words = [[bit for bit in word] for word in words]

    # assume only hmms of one state size
    states = list(range(complexity))
   
    for index in start_indices:
        # read in the hmm data
        t_values = hmms[index+1:index+complexity+1, 1:complexity+1]
        e_values = hmms[index+complexity+2:index+2*complexity+2, 1:size+1]
        pi = ProbabilityVector({state: hmms[index+2*complexity+3, state+1] for state in states})

        # initialize the hmm
        T = ProbabilityMatrix.initialize(states, observables)
        E = ProbabilityMatrix.initialize(states, observables)
        T.values, E.values = t_values, e_values
        hml = HiddenMarkovLayer(T, E, pi)

        if hml.score(word) > max([hml.score(word) for word in words]) - epsilon:
            print('HMM witnessing complexity = {} for word {}:'.format(complexity, ''.join(word)))
            print('transmission matrix\n', hml.T.df, '\n\nemission matrix\n', hml.E.df, '\n\ninitial state probability\n', hml.pi.df, '\n')
            print('emission probability: ', hml.score(word))
            print('stopping early')
            return 1

    return 0