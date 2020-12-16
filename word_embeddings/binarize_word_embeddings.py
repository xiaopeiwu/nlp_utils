"""
Script to convert a word embedding text file (such as GloVe, word2vec) to a binary format to feed to models

Usage:
    binarize_word_embeddings.py <embedding_file_path>

Arguments:
    <embedding_file_path>        the input word embedding text file
"""

from docopt import docopt
import numpy as np


def main():
    args = docopt(__doc__)

    embedding_file_path = args["<embedding_file_path>"]
    print("Embedding text file provided: {}".format(embedding_file_path))

    # load the embeddings from file
    vecs, vocab = load_embeddings(embedding_file_path)

    # output vocab file and binary file
    out_bin_file_name = embedding_file_path.rstrip(".txt")
    out_vocab_file_path = out_bin_file_name + ".vocab"

    print("Saving vocabulary file to: {}".format(out_vocab_file_path))
    with open(out_vocab_file_path, "w", encoding='utf=8') as f_out:
        f_out.write("\n".join(vocab))
        f_out.write("\n")

    print("Saving binary file to: {}".format(out_bin_file_name + ".npy"))
    np.save(out_bin_file_name, vecs)  # default is to use allow_pickle=False when saving arrays instead of objects


def load_embeddings(embedding_file):
    """
    Given a word embedding text file, load the embeddings and index the vocab
    :param embedding_file: embedding file in text format
    :return vectors (nparray) and list of vocab
    """
    vectors = []
    vocab = []
    with open(embedding_file, encoding='utf=8') as f_in:
        first_line = f_in.readline().strip()
        dim = len(first_line.split()) - 1
        print("Embeddings have {} dimensions".format(dim))

        counter = 0
        for line in f_in:
            if len(line.split()) - 1 == dim:
                parts = line.strip().split()
                vocab.append(parts[0])
                vectors.append([float(x) for x in parts[1:]])
                counter += 1
        print("Embedding file has a vocabulary size of {}".format(counter))

    vectors = np.array(vectors, dtype=np.float64)
    # https://numpy.org/doc/stable/user/basics.types.html
    # np.float64 / np.float_ (C type double) - Note that this matches the precision of the builtin python float
    return vectors, vocab


if __name__ == "__main__":
    main()



