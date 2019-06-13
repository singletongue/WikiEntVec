# Wikipedia Entity Vectors


## Introduction

**Wikipedia Entity Vectors** [1] is a distributed representation of words and named entities (NEs).
The words and NEs are mapped to the same vector space.
The vectors are trained with skip-gram algorithm using preprocessed Wikipedia text as the corpus.


## Downloads

Pre-trained vectors are downloadable from the [Releases](https://github.com/singletongue/WikiEntVec/releases) page.

Several old versions are available at [this site](http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/).


## Specs

Each version of the pre-trained vectors contains three files: `word_vectors.txt`, `entity_vectors.txt`, and `all_vectors.txt`.
These files are text files in word2vec output format, in which the first line declares the vocabulary size and the vector size (number of dimensions), followed by lines of tokens and their corresponding vectors.

`word_vectors.txt` and `entity_vectors.txt` contains vectors for words and NEs, respectively.
For `entity_vectors.txt`, white spaces within names of NEs are replaced with underscores like `United_States`.
`all_vectors.txt` contains vectors of both words and embeddings in one file, where each NE token is formatted with square brackets like `##United_States##`

Pre-trained vectors are trained under the configurations below (see Manual training for details):

### `make_corpus.py`

#### Japanese

We used [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) (v0.0.6) for tokenizing Japanese texts.

|Option              |Value                                                 |
|:-------------------|:-----------------------------------------------------|
|`--cirrus_file`     |path to `jawiki-20190520-cirrussearch-content.json.gz`|
|`--output_file`     |path to the output file                               |
|`--tokenizer`       |`mecab`                                               |
|`--tokenizer_option`|`-d <directory of mecab-ipadic-NEologd dictionary>`   |

### `train.py`

|Option         |Value                                           |
|:--------------|:-----------------------------------------------|
|`--corpus_file`|path to a corpus file made with `make_corpus.py`|
|`--output_dir` |path to the output directory                    |
|`--embed_size` |`100`, `200`, or `300`                          |
|`--window_size`|`10`                                            |
|`--sample_size`|`10`                                            |
|`--min_count`  |`10`                                            |
|`--epoch`      |`5`                                             |
|`--workers`    |`20`                                            |


## Manual training

You can manually process Wikipedia dump file and train a skip-gram model on the preprocessed file.


### Requirements

- Python 3.6
- gensim
- logzero
- MeCab and its Python binding (mecab-python3) (optional: required for tokenizing Japanese texts)


### Steps

1. Download Wikipedia Cirrussearch dump file from [here](https://dumps.wikimedia.org/other/cirrussearch/).
    - Make sure to choose a file named like `**wiki-YYYYMMDD-cirrussearch-content.json.gz`.
2. Clone this repository.
3. Preprocess the downloaded dump file.
    ```
    $ python make_corpus.py --cirrus_file <dump file> --output_file <corpus file>
    ```
    If you're processing Japanese version of Wikipedia, make sure to use MeCab tokenizer by setting `--tokenizer mecab` option.
    Otherwise, the text will be tokenized by a simple rule based on regular expression.
4. Train the model
    ```
    $ python train.py --corpus_file <corpus file> --output_dir <output directory>
    ```

    You can configure options below for training a model.

    ```
    usage: train.py [-h] --corpus_file CORPUS_FILE --output_dir OUTPUT_DIR
                    [--embed_size EMBED_SIZE] [--window_size WINDOW_SIZE]
                    [--sample_size SAMPLE_SIZE] [--min_count MIN_COUNT]
                    [--epoch EPOCH] [--workers WORKERS]

    optional arguments:
      -h, --help            show this help message and exit
      --corpus_file CORPUS_FILE
                            Corpus file (.txt)
      --output_dir OUTPUT_DIR
                            Output directory to save embedding files
      --embed_size EMBED_SIZE
                            Dimensionality of the word/entity vectors [100]
      --window_size WINDOW_SIZE
                            Maximum distance between the current and predicted
                            word within a sentence [5]
      --sample_size SAMPLE_SIZE
                            Number of negative samples [5]
      --min_count MIN_COUNT
                            Ignores all words/entities with total frequency lower
                            than this [5]
      --epoch EPOCH         number of training epochs [5]
      --workers WORKERS     Use these many worker threads to train the model [2]
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

For instance, if an article has a hyperlink to "Mercury (planet)" assigned to the anchor text "Mercury", we replace all the other appearances of "Mercury" in the same article with the special token `##Mercury_(planet)##`.

Note that the diversity of NE mentions is resolved by replacing possibly diverse anchor texts with special tokens which are unique to NEs.
Moreover, the ambiguity of NE mentions is also addressed by making "one sense per discourse" assumption; we assume that NEs mentioned by possibly ambiguous phrases can be determined by the context or the document.
With this assumption, the phrases "Mercury" in the above example are neither replaced with `##Mercury_(element)##` nor `##Freddie_Mercury##`, since the article does not have such mentions as hyperlinks.

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
