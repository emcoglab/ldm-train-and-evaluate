import re
import nltk

_treebank_word_tokenizer = nltk.tokenize.TreebankWordTokenizer()

# copied from word_tokenize
improved_open_quote_regex = re.compile(u'([«“‘])', re.U)
improved_close_quote_regex = re.compile(u'([»”’])', re.U)
improved_punct_regex = re.compile(r'([^\.])(\.)([\]\)}>"\'' u'»”’ ' r']*)\s*$', re.U)

# Add extra quotes and currency symbols
improved_currency_regex = re.compile(r'[£€¥]')

_treebank_word_tokenizer.STARTING_QUOTES.insert(0, (improved_open_quote_regex, r' \1 '))
_treebank_word_tokenizer.ENDING_QUOTES.insert(0, (improved_close_quote_regex, r' \1 '))
_treebank_word_tokenizer.PUNCTUATION.insert(0, (improved_punct_regex, r'\1 \2 \3 '))
_treebank_word_tokenizer.PUNCTUATION.insert(0, (improved_currency_regex, r' \g<0> '))


def modified_word_tokenize(text, language='english', preserve_line=False):
    """
    A modified version of NLTK's recommended word tokenizer
    The tokenizer has been modified to treat currency symbols £, €, ¥ in the same way as $.

    :param text: text to split into words
    :param text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
    :type preserve_line: bool
    """
    sentences = [text] if preserve_line else nltk.tokenize.sent_tokenize(text, language)
    return [token for sent in sentences
            for token in _treebank_word_tokenizer.tokenize(sent)]
