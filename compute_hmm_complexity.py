import argparse, sys
from hmm import HiddenMarkovLayer, HiddenMarkovModel
from get_words import get_all_words, get_unique_words
from verify_hmm_complexity import verify_hmm_complexity

def compute_complexity(word, size=2, hmm_searches=1000, epochs=100, epoch_searches=10, epsilon=0, verify=False, proportion=0.5, print_witness=True):
    observables = [str(n) for n in range(size)]

    # by definition
    if not word:
        return 0

    # lemma 7.31
    if len(set(word)) == 1:
        return 1

    # get the set of all other binary words of length(word)
    words = get_all_words(len(word), size)
    words.remove(word)

    # reformat to fit the hmm implementation
    word = [bit for bit in word]
    words = [[bit for bit in word] for word in words]

    complexity = 2
    while(True):
        print('Searching through HMMs with {} states'.format(complexity))

        # store hmms with currnet number of states in a txt file
        if verify:
            f = open('{}_verification_complexity_{}_size_{}.txt'.format(''.join(word), complexity, size), 'w')

        for _ in range(hmm_searches):
            # initialize the hmm
            states = list(range(complexity))
            hml = HiddenMarkovLayer.initialize(states, observables)
            hmm = HiddenMarkovModel(hml)

            # train and score
            hmm.train(word, epochs)
            score = hml.score(word)
            probs = [hml.score(word) for word in words]

            # if this hmm doesn't witness complexity
            if not (score > max(probs) - epsilon):
                # train the hmm on the given word again
                for _ in range(epoch_searches):
                    if not (score > max(probs) - epsilon):
                        hmm.train(word, 1)

                if verify:
                    # transmission matrix has dimension (complexity) X (complexity), emission matrix has dimension (complexity) X (size)
                    # so add dummy columns to the narrower matrix so that the txt file can be read as a numpy array
                    temp_T_df = hml.T.df
                    temp_E_df = hml.E.df
                    temp_pi_df = hml.pi.df

                    if size < complexity:
                        for i in range(size, complexity):
                            dummy_col_name = 'dummy_{}'.format(i)
                            temp_E_df[dummy_col_name] = -1

                    elif complexity < size:
                        # in this case, also pad the probability vector
                        for i in range(complexity, size):
                            dummy_col_name = 'dummy_{}'.format(i)
                            temp_T_df[dummy_col_name] = -1
                            temp_pi_df[dummy_col_name] = -1

                    f.write('-1'+temp_T_df.to_string()+'\n')
                    f.write('0'+temp_E_df.to_string()+'\n')
                    f.write('0'+temp_pi_df.to_string()+'\n')

                
            # if the hmm doesn't witness complexity by this point, move on
            if score > max(probs) - epsilon:
                if print_witness:
                    print('HMM witnessing complexity = {} for word {}:'.format(complexity, ''.join(word)))
                    print('transmission matrix\n', hml.T.df, '\n\nemission matrix\n', hml.E.df, '\n\ninitial state probability\n', hml.pi.df, '\n')
                    print('emission probability: ', hml.score(word))

                if verify:
                    f.close()
                return complexity

        if verify:
            f.close()
            if verify_hmm_complexity(word, size, complexity, proportion) == 1:
                break
            else:
                print('HMMs of complexity {} verified succesfully'.format(complexity))

        complexity += 1

def parse_compute_complexity(word, verify, proportion, size):
    if type(word) == list and len(word) > 1:
        sys.exit('enter a single string as argument')

    if not proportion:
        proportion = 0.5
    else:
        proportion = proportion[0]

    compute_complexity(word, size=size, verify=verify, proportion=proportion)

def compute_all_complexity(n, size):
    if len(n) > 1:
        sys.exit('enter a single natural number as argument')

    words = get_unique_words(int(n[0]), size)
    for word in words:
        compute_complexity(word, size=size)
        print('################################################################')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word',
                        help='flag to compute HMM complexity of a given word' )                        

    parser.add_argument('--all',
                        help='flag to compute HMM complexity of all words of a given length')

    parser.add_argument('--verify',
                        action='store_true',
                        help='flag to verify that no HMM searched with fewer states witnessed complexity')

    parser.add_argument('verify_input', type=float, nargs='*', help='proportion of HMMs to verify')                        

    parser.add_argument('--size',
                        help='flag to set size of the language outputted by the HMMs')

    args = parser.parse_args()

    if not args.size:
        size = 2
    else:
        size = int(args.size)

    if args.word:
        parse_compute_complexity(args.word, args.verify, args.verify_input, size)
    elif args.all:
        compute_all_complexity(args.all, size)
    else:
        print('use either --word or --all to use script')