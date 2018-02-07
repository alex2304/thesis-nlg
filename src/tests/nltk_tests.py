from os import environ
from os.path import join, dirname

import nltk
from nltk import pos_tag_sents, SnowballStemmer, PorterStemmer, LancasterStemmer, WordNetLemmatizer, sent_tokenize, \
    word_tokenize, RecursiveDescentParser, pos_tag, str2tuple, flatten, tuple2str
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer, PunktSentenceTokenizer

from src.nltk_utils import penn_to_wn, Tokenizer, create_grammar, Stemmer, Lemmatizer
from src.io import load_corpora


def nltk_download(res_id: str):
    return nltk.download(res_id)


# 1) Sentence segmentation (
# 2) Tokenization   WordPunctTokenizer, TreebankWordTokenizer(def)
# 3) POS tagging    maxent_treebanck_pos_tagger(def),
# 4) Stemming tokens - finding stems of tokens
# 5) Lemmatizing
# 6) Sentences parsing

def test_all():
    text = 'It is going so bad. Have you\'d any problems yet? Yeah!!! Good bye!'
    print(text, '\n')

    # 1
    s_tokenizer = PunktSentenceTokenizer(text)
    sentences = s_tokenizer.tokenize(text)
    print('%d sentences' % len(sentences))
    print(sentences, '\n')

    # 2
    tokenizer = WordPunctTokenizer()
    sentences_tokens = tokenizer.tokenize_sents(sentences)
    print('%d tokens' % len(sentences_tokens))
    print(sentences_tokens, '\n')
    tokenizer = TreebankWordTokenizer()
    sentences_tokens = tokenizer.tokenize_sents(sentences)
    print('%d tokens' % len(sentences_tokens))
    print(sentences_tokens, '\n')

    # 3
    tagged_sentences = pos_tag_sents(sentences_tokens)
    print(tagged_sentences, '\n')

    # 4
    def stem(stemmer):
        stems = [[stemmer.stem(token) for token in s_tokens]
                 for s_tokens in sentences_tokens]
        print(stemmer.__class__.__name__)
        print(stems, '\n')

    stemmers = [PorterStemmer(), SnowballStemmer('english', ignore_stopwords=True), LancasterStemmer()]
    for s in stemmers:
        stem(s)

    # 5
    lemmatizer = WordNetLemmatizer()
    lemmas = [[lemmatizer.lemmatize(t_t[0], penn_to_wn(t_t[1])) for t_t in t_s]
              for t_s in tagged_sentences]
    print(lemmas)

    # 6
    from os.path import join
    _path = join(__file__, './standford_parser')
    parser = StanfordParser(join(_path, 'stanford-parser.jar'), join(_path, 'stanford-parser-3.4.1-models.jar'))
    iter_trees = list(parser.tagged_parse_sents(tagged_sentences))
    for _iter in iter_trees:
        try:
            tree = next(_iter)
            tree.draw()
        except StopIteration:
            pass


def test_parser():
    environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_91'

    _path = join(dirname(__file__), './standford_parser')
    parser = StanfordParser(join(_path, 'stanford-parser.jar'), join(_path, 'stanford-parser-3.4.1-models.jar'))

    def subtree_filter(t):
        h = t.height()
        l = len(t.leaves())

        if h <= 2:
            return True
        elif 2 <= l <= 3:
            return True
        else:
            return False

    while True:
        text = input('> ')

        sents = sent_tokenize(text)
        sents_tokens = [word_tokenize(s) for s in sents]
        tagged_sentences = pos_tag_sents(sents_tokens)

        trees_iters = list(parser.tagged_parse_sents(tagged_sentences))
        for tree_iter in trees_iters:
            try:
                tree = next(tree_iter)

                for subtree in tree.subtrees(subtree_filter):
                    if subtree.label() in ('NP', 'VP'):
                        print(subtree.productions())

                    print(subtree.leaves())

                tree.draw()
            except StopIteration:
                pass


def test_tokenizer():
    tokenizer = Tokenizer()
    # tokenizer = TreebankWordTokenizer()

    while True:
        print(tokenizer.tokenize(input('> ')))


def test_pos():
    tokenizer = Tokenizer()

    while True:
        print(pos_tag(tokenizer.tokenize(input('> '))))


def test_cfg():
    sents = 10
    t = Tokenizer()
    l = Lemmatizer()
    s = Stemmer()

    text = load_corpora().lower()

    ttokens = [[tuple2str(tt) for tt in s_tts]
               for s_tts in pos_tag_sents(t.tokenize_sents(sent_tokenize(text)[:sents]))]

    ttokens = [str2tuple(tt) for tt in flatten(ttokens)]

    g = create_grammar(ttokens)
    p = RecursiveDescentParser(g)

    while True:
        try:
            tokens = t.tokenize(input('> ').lower())
            ttokens = pos_tag(tokens)

            stems = [s.stem(t) for t in tokens]
            lemmas = [l.lemmatize(token, tag) for token, tag in ttokens]

            terms = [tag for token, tag in ttokens]

            print(terms)
            # p.trace(2)

            for tree in p.parse(terms):
                tree.draw()
                break
            else:
                print('no trees')
        except ValueError as e:
            print(e)


def test_kneser_ney():
    def train_and_test(est):
        hmm = trainer.train_supervised(train_corpus, estimator=est)

        print('%.2f%%' % (100 * hmm.evaluate(test_corpus)))

    corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:500]

    est = nltk.KneserNeyProbDist(nltk.FreqDist((tt for sent in corpus for tt in nltk.trigrams(sent))))

    print(est.prob((('right', 'RB'), ("''", "''"), (',', ','))))
    # print(est.max())
    print(est.generate())
    # print(est.sa mples())
    return
    corpus = [[((x[0], y[0], z[0]), (x[1], y[1], z[1]))
               for x, y, z in nltk.trigrams(sent)]
              for sent in corpus]
    tag_set = nltk.unique_list(tag for sent in corpus for (word, tag) in sent)
    print(len(tag_set))

    symbols = nltk.unique_list(word for sent in corpus for (word, tag) in sent)
    print(len(symbols))

    trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)

    train_corpus, test_corpus = [], []
    for i in range(len(corpus)):
        if i % 5:
            train_corpus += [corpus[i]]
        else:
            test_corpus += [corpus[i]]

    print(len(train_corpus), len(test_corpus))

    kn = lambda fd, bins: nltk.KneserNeyProbDist(fd, bins)

    train_and_test(kn)


if __name__ == '__main__':
    # test_parser()
    # test_tokenizer()
    # test_cfg()
    # test_pos()
    # test_kneser_ney()
    test_parser()
