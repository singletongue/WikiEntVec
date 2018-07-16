# coding=utf-8

import argparse
import logging
from pathlib import Path

from gensim.models.word2vec import LineSentence, Word2Vec


logging.basicConfig(level=logging.INFO, datefmt='%m/%d %H:%M:%S',
    format='[%(asctime)s] %(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_file', type=str,
        help='corpus file')
    parser.add_argument('out_dir', type=str,
        help='output directory to store embedding files')
    parser.add_argument('--size', type=int, default=100,
        help='size of word vectors [100]')
    parser.add_argument('--window', type=int, default=5,
        help='maximum window size [5]')
    parser.add_argument('--mode', choices=('cbow', 'sg'), default='cbow',
        help='training algorithm: '
             '"cbow" (continuous bag of words) or "sg" (skip-gram) [cbow]')
    parser.add_argument('--loss', choices=('ns', 'hs'), default='ns',
        help='loss function: '
             '"ns" (negative sampling) or "hs" (hierarchical softmax) [ns]')
    parser.add_argument('--sample', type=float, default=1e-3,
        help='threshold of frequency of words to be down-sampled [1e-3]')
    parser.add_argument('--negative', type=int, default=5,
        help='number of negative examples [5]')
    parser.add_argument('--threads', type=int, default=2,
        help='number of worker threads to use for training [2]')
    parser.add_argument('--iter', type=int, default=5,
        help='number of iterations in training [5]')
    parser.add_argument('--min_count', type=int, default=5,
        help='discard all words with total frequency lower than this [5]')
    parser.add_argument('--alpha', type=float, default=0.025,
        help='initial learning rate [0.025]')
    args = parser.parse_args()

    if not Path(args.out_dir).exists():
        Path(args.out_dir).mkdir()

    fname_word_vectors = Path(args.out_dir) / 'word_vectors.txt'
    fname_entity_vectors = Path(args.out_dir) / 'entity_vectors.txt'
    fname_all_vectors = Path(args.out_dir) / 'all_vectors.txt'

    settings = {
        'size': args.size,
        'alpha': args.alpha,
        'window': args.window,
        'min_count': args.min_count,
        'sample': args.sample,
        'workers': args.threads,
        'sg': int(args.mode == 'sg'),
        'hs': int(args.loss == 'hs'),
        'negative': args.negative,
        'iter': args.iter
    }
    for (key, value) in settings.items():
        logging.info(f'{key}: {value}')

    logging.info('training the model')
    model = Word2Vec(sentences=LineSentence(args.corpus_file), **settings)

    word_vocab_size = 0
    entity_vocab_size = 0
    for token in model.wv.vocab:
        if token.startswith('[') and token.endswith(']'):
            entity_vocab_size += 1
        else:
            word_vocab_size += 1

    all_vocab_size = word_vocab_size + entity_vocab_size

    logging.info('writing embeddings to a file')
    with open(fname_word_vectors, 'w') as fw, \
            open(fname_entity_vectors, 'w') as fe, \
            open(fname_all_vectors, 'w') as fa:
        print(word_vocab_size, args.size, file=fw)
        print(entity_vocab_size, args.size, file=fe)
        print(all_vocab_size, args.size, file=fa)
        for (token, _) in sorted(
                model.wv.vocab.items(), key=lambda t: -t[1].count):
            vector = model.wv[token]
            if token.startswith('[') and token.endswith(']'):
                print(token[1:-1], *vector, file=fe)
            else:
                print(token, *vector, file=fw)

            print(token, *vector, file=fa)


if __name__ == '__main__':
    main()
