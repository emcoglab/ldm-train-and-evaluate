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
    def __init__(self, metadata: CorpusMetaData, batch_size: int):
        """

        :type batch_size: int
        :type metadata: CorpusMetaData
        :param metadata:
        :param batch_size:
        Size of batch
        """
        self.metadata = metadata
        self.batch_size = batch_size

    # TODO: does using disjoint "sentences" here lead to unpleasant edge effects?
    def __iter__(self):
        batch = []
        with open(self.metadata.path, mode="r", encoding="utf-8") as corpus_file:
            for token in corpus_file:
                batch.append(token.strip())
                if len(batch) >= self.batch_size:
                    yield batch
                    # Empty the batch to be refilled
                    batch = []
            # Don't forget the last batch, if there is one
            if len(batch) > 0:
                yield batch
