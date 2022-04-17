import numpy as np
import argparse, sys
from get_words import get_all_words
from hmm import ProbabilityVector, ProbabilityMatrix, HiddenMarkovLayer

def construct_0n1_witness(n):
    states = list(range(n))
    observables = ['0', '1']

    pi = ProbabilityVector({state: 1 if state == 0 else 0 for state in states})
    t_values = np.zeros((n, n))
    e_values = np.zeros((n, len(observables)))

    for i in range(n-1):
        t_values[i][i] = 1/(n+1)
        t_values[i][i+1] = n/(n+1)
        e_values[i][0] = 1

    t_values[n-1][n-1] = 1
    e_values[n-1][0] = 1/(n+1)
    e_values[n-1][1] = n/(n+1)

    T = ProbabilityMatrix.initialize(states, states)
    E = ProbabilityMatrix.initialize(states, observables)
    T.values, E.values = t_values, e_values

    hml = HiddenMarkovLayer(T, E, pi)
    return hml

def print_witness_scores(n, states):
    hml = construct_0n1_witness(states)
    words = get_all_words(n, 2)
    scores = {word: hml.score(word) for word in words}

    for word, score in scores.items():
        print('word: ', word)
        print('score: ', score)

    max_score_word = [word for word in scores.keys() if scores[word] == max(scores.values())][0]
    print('most probable word was', max_score_word, 'with score', scores[max_score_word])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',
                        help='flag that sets the value of n in 0^n1')
    parser.add_argument('--states',
                        help='flag that sets the number of states in the witnessing HMM')

    args = parser.parse_args()

    if not args.n or not args.states:
        sys.exit('enter values to set the length of the word and the number of states')
    
    print_witness_scores(int(args.n), int(args.states))