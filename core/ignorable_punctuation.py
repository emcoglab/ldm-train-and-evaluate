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
#       …
#       –
#       ‘
#       ’
ignorable_punctuation = r"""!"#'()*,-./:;<>?[\]^_`{|}~…–‘’"""
