import nltk.corpus.reader.bnc

# full BNC text corpus
a = nltk.corpus.reader.bnc.BNCCorpusReader(
    root='/Users/caiwingfield/Box Sync/LANGBOOT Project/Corpus Analysis/BNC/XML version/Texts',
    fileids=r'[A-K]/\w*/\w*\.xml')

# how many sentences
len(a.sents())

# how many words
len(a.words())
