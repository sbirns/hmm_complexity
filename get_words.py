from itertools import product

def get_all_words(n, size):
    # returns all  words of length n, over an alphabet of size (size)
    letters = [str(char) for char in list(range(size))]
    
    return [''.join(x) for x in product(letters, repeat=n)]

def get_unique_words(n, size):
    # returns all unique (up to some permutation) words of length n, over an alphabet of size (size)
    words = get_all_words(n, size)

    letters = [str(char) for char in list(range(size))]
    mappings = list(product(letters, repeat=2))

    for word in words:
        for map in mappings:
            if len(list(set(map))) > 1 and word.replace(map[0], map[1]) in words:
                words.remove(word.replace(map[0], map[1]))
    return words