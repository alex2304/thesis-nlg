import nltk
from nltk import pos_tag_sents, SnowballStemmer, PorterStemmer, LancasterStemmer, WordNetLemmatizer, sent_tokenize, \
    word_tokenize, RegexpTokenizer
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer, PunktSentenceTokenizer

from src.nltk_utils import penn_to_wn, Tokenizer


def polyglot_download(module_id: str):
    _dir = './data'
    from polyglot.downloader import downloader
    return downloader.download(module_id, download_dir=_dir)


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
    from os.path import join, dirname
    from os import environ
    environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_91'

    _path = join(dirname(__file__), './standford_parser')
    parser = StanfordParser(join(_path, 'stanford-parser.jar'), join(_path, 'stanford-parser-3.4.1-models.jar'))

    # tagged sentences

    while True:
        text = input('> ')

        sents = sent_tokenize(text)
        sents_tokens = [word_tokenize(s) for s in sents]
        tagged_sentences = pos_tag_sents(sents_tokens)

        trees_iters = list(parser.tagged_parse_sents(tagged_sentences))
        for tree_iter in trees_iters:
            try:
                tree = next(tree_iter)

                # def subtree_filter(t):
                #     h = t.height()
                #     l = len(t.leaves())
                #
                #     if h <= 2:
                #         return True
                #     elif 2 <= l <= 3:
                #         return True
                #     else:
                #         return False
                #
                # for subtree in tree.subtrees(subtree_filter):
                #     print(subtree.leaves())

                tree.draw()
            except StopIteration:
                pass


def test_tokenizer():
    tokenizer = Tokenizer()
    # tokenizer = TreebankWordTokenizer()

    while True:
        print(tokenizer.tokenize(input('> ')))

if __name__ == '__main__':
    # test_parser()
    test_tokenizer()