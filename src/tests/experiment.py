from nltk import ConditionalFreqDist, FreqDist
from nltk.corpus import brown, stopwords

if __name__ == '__main__':
    stop_words = set(stopwords.words('english'))

    words = [(token.lower(), tag)
             for token, tag in brown.tagged_words(tagset='universal')]

    ttokens = words

    cond_freq_dist = ConditionalFreqDist(
        (tag, token) for token, tag in ttokens
    )

    for c in cond_freq_dist.conditions():
        print(c, cond_freq_dist[c].most_common(10))

    freq_dist = FreqDist([tag for _, tag in ttokens])

    print(freq_dist.most_common(10))