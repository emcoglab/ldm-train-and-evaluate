# string.punctuation  = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# same as above except:
#  - we don't want to ignore:
#       $
#       %
#       &
#       @
#       +
#       =
#  - we do want to ignore
#       \u2026 …
#       \u2013 –
ignorable_punctuation = r"""!"#'()*,-./:;<>?[\]^_`{|}~\u2026\u2013"""
