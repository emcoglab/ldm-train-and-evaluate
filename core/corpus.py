class CorpusMetaData:
    """
    Corpus metadata
    """

    def __init__(self, name, path, info_path=None):
        self.name = name
        self.path = path
        self.info_path = info_path


class BatchedCorpus(object):
    """
    Corpus which yields batches of tokens
    """
    def __init__(self, filename, batch_size: int):
        """
        :param filename: Location of corpus file
        :param batch_size: Size of batch
        """
        self.filename = filename
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        with open(self.filename, mode="r", encoding="utf-8") as corpus_file:
            for token in corpus_file:
                batch.append(token.strip())
                if len(batch) >= self.batch_size:
                    yield batch
                    # Empty the batch to be refilled
                    batch = []
            # Don't forget the last batch, if there is one
            if len(batch) > 0:
                yield batch
