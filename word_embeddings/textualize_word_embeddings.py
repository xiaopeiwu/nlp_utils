"""
Script to manually convert a word embedding binary file to a word2vec type of text format using Gensim.
Exception: fastText - Use the fastText API for fT's binary file!

Usage:
    textualize_word_embeddings.py <embedding_file_path>

Arguments:
    <embedding_file_path>        the input word embedding binary file
"""

import gensim
import docopt


def main():
    args = docopt.docopt(__doc__)
    embedding_file_path = args["<embedding_file_path>"]
    print("Embedding binary file provided: {}".format(embedding_file_path))

    # load binary embeddings using gensim's KeyedVectors
    vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)

    # write to a text file
    out_text_file_path = embedding_file_path.rstrip(".bin") + ".txt"
    with open(out_text_file_path, "w", encoding="utf-8") as f_out:
        for word in vectors.index2word:
            vector = " ".join(map(str, list(vectors[word])))
            f_out.write(word + ' ' + vector + '\n')
            f_out.write("\n")


if __name__ == '__main__':
    main()
