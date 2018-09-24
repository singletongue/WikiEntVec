# Wikipedia Entity Vectors


## Introduction

**Wikipedia Entity Vectors** [1] is a distributed representation of words and named entities (NEs).
The words and NEs are mapped to the same vector space.
The vectors are trained with skip-gram algorithm using preprocessed Wikipedia text as the corpus.


## Downloads

Pre-trained vectors are downloadable from the Releases page.

Several old versions are available at [this site](http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/).


## Specs

Each version of the pre-trained vectors contains three files: `word_vectors.txt`, `entity_vectors.txt`, and `all_vectors.txt`.
These files are text files in word2vec output format, in which the first line declares the vocabulary size and the vector size (number of dimensions), followed by lines of tokens and their corresponding vectors.

`word_vectors.txt` and `entity_vectors.txt` contains vectors for words and NEs, respectively.
For `entity_vectors.txt`, white spaces within names of NEs are replaced with underscores like `United_States`.
`all_vectors.txt` contains vectors of both words and embeddings in one file, where each NE token is formatted with square brackets like `[United_States]`

Pre-trained vectors are trained under the configurations below (see Manual training for details):

### `generate_corpus.py`

#### Japanese

|Option       |Value  |
|:------------|:------|
|`--tokenizer`|`mecab`|

<!-- #### English

|Option       |Value      |
|:------------|:----------|
|`--tokenizer`|`regexp`   |
|`--lower`    |(specified)| -->

### `train.py`

|Option       |Value                  |
|:------------|:----------------------|
|`--size`     |`100`, `200`, and `300`|
|`--window`   |`10`                   |
|`--mode`     |`sg`                   |
|`--loss`     |`ns`                   |
|`--sample`   |`1e-3`                 |
|`--negative` |`10`                   |
|`--threads`  |`20`                   |
|`--iter`     |`5`                    |
|`--min_count`|`10`                   |
|`--alpha`    |`0.025`                |


## Manual training

You can manually process Wikipedia dump file and train the CBOW or skip-gram model on the preprocessed file.


### Requirements

- Python 3.6
- gensim
- MeCab and its Python binding (mecab-python3) (optional: required for tokenizing text in Japanese)


### Steps

1. Download Wikipedia Cirrussearch dump file from [here](https://dumps.wikimedia.org/other/cirrussearch/).
    - Make sure to choose a file with a name like `**wiki-YYYYMMDD-cirrussearch-content.json.gz`.
2. Clone this repository.
3. Preprocess the downloaded dump file.
    ```
    $ python generate_corpus.py path/to/dump/file.json.gz path/to/output/corpus/file.txt.bz2
    ```
    If you're processing Japanese version of Wikipedia, make sure to use MeCab tokenizer by setting `--tokenizer mecab` option.
    Otherwise, the text will be tokenized by a simple rule based on regular expression.
4. Train the model
    ```
    $ python train.py path/to/corpus/file.txt.bz2 path/to/output/directory/ --size 100 --window 10 --mode sg --loss ns --sample 1e-3 --negative 10 --threads 20 --iter 5 --min_count 10 --alpha 0.025
    ```

    You can configure options below for training a model.

    ```
    usage: train.py [-h] [--size SIZE] [--window WINDOW] [--mode {cbow,sg}]
                    [--loss {ns,hs}] [--sample SAMPLE] [--negative NEGATIVE]
                    [--threads THREADS] [--iter ITER] [--min_count MIN_COUNT]
                    [--alpha ALPHA]
                    corpus_file out_dir

    positional arguments:
    corpus_file           corpus file
    out_dir               output directory to store embedding files

    optional arguments:
    -h, --help            show this help message and exit
    --size SIZE           size of word vectors [100]
    --window WINDOW       maximum window size [5]
    --mode {cbow,sg}      training algorithm: "cbow" (continuous bag of words)
                          or "sg" (skip-gram) [cbow]
    --loss {ns,hs}        loss function: "ns" (negative sampling) or "hs"
                          (hierarchical softmax) [ns]
    --sample SAMPLE       threshold of frequency of words to be down-sampled
                          [1e-3]
    --negative NEGATIVE   number of negative examples [5]
    --threads THREADS     number of worker threads to use for training [2]
    --iter ITER           number of iterations in training [5]
    --min_count MIN_COUNT
                          discard all words with total frequency lower than this
                          [5]
    --alpha ALPHA         initial learning rate [0.025]
    ```


## Concepts

There are several methods to learn distributed representations (or embeddings) of words, such as CBOW and skip-gram [2].
These methods train a neural network to predict contextual words given a word in a sentence from large corpora in an unsupervised way.

However, there are a couple of problems when applying these methods to learning distributed representations of NEs.
One problem is that many NEs consist of multiple words (such as "New York" and "George Washington"), which makes a simple tokenization of text undesirable.

Other problems are the diversity and ambiguity of NE mentions.
For each NE, several expression can be used to mention the NE.
For example, "USA", "US", "United States", and "America" can all express the same country.
On the other hand, the same words and phrases can refer to different entities.
For example, the word "Mercury" may represent a planet or an element or even a person (such as "Freddie Mercury", the vocalist for the rock group Queen).
Therefore, in order to learn distributed representations of NEs, one must identify the spans of NEs in the text and recognize the mentioned NEs so that they are not treated just as a sequence of words.

To address these problems, we used Wikipedia as the corpus and utilized its internal hyperlinks for identifying mentions of NEs in article text.

For each article in Wikipedia, we performed the following preprocessing.

First, we extracted all hyperlinks (pairs of anchor text and the linked article) from the source text (a.k.a wikitext) of an article.

Next, for each hyperlink, we replaced the appearances of anchor text in the article body with special tokens representing the linked articles.

For instance, if an article has a hyperlink to "Mercury (planet)" assigned to the anchor text "Mercury", we replace all the other appearances of "Mercury" in the same article with the special token `[Mercury_(planet)]`.

Note that the diversity of NE mentions is resolved by replacing possibly diverse anchor texts with special tokens which are unique to NEs.
Moreover, the ambiguity of NE mentions is also addressed by making "one sense per discourse" assumption; we assume that NEs mentioned by possibly ambiguous phrases can be determined by the context or the document.
With this assumption, the phrases "Mercury" in the above example are neither replaced with `[Mercury_(element)]` nor `[Freddie_Mercury]`, since the article does not have such mentions as hyperlinks.

We used the preprocessed Wikipedia articles as the corpus and applied skip-gram algorithm to learn distributed representations of words and NEs.
This means that words and NEs are mapped to the same vector space.


## Licenses

The pre-trained vectors are distributed under the terms of the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

The codes in this repository are distributed under the MIT License.


## References

[1] Masatoshi Suzuki, Koji Matsuda, Satoshi Sekine, Naoaki Okazaki and Kentaro
Inui. A Joint Neural Model for Fine-Grained Named Entity Classification of
Wikipedia Articles. IEICE Transactions on Information and Systems, Special
Section on Semantic Web and Linked Data, Vol. E101-D, No.1, pp.73-81, 2018.

[2] Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. Efficient Estimation
of Word Representations in Vector Space. ICLR, 2013.

[3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, Jeff Dean.
Distributed Representations of Words and Phrases and their Compositionality.
NIPS, 2013.


## Acknowledgments

This work was partially supported by Research and Development on Real World Big Data Integration and Analysis.
