class CorpusMetaData:
    """
    Corpus metadata
    """

    def __init__(self, name, path, info_path=None):
        self.name = name
        self.path = path
        self.info_path = info_path


class SourceTargetPair:
    """
    A pair of a source directory and a target directory.
    """

    def __init__(self, source: CorpusMetaData, target: CorpusMetaData):
        self.source = source
        self.target = target
