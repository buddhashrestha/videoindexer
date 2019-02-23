import faiss


class FaissWrapper(object):
    def __init__(self, dim=128, clusters=1):
        self.dim = dim
        self.nlist = 1
        self.clusters = clusters
        self.quantizer = faiss.IndexFlatL2(self.dim)  # the other index
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.dim, self.nlist, faiss.METRIC_L2)

    def train(self, data):
        self.index.train(data)
        self.index.add(data)

    def searcher(self, query):
        D, I = self.index.search(query, self.clusters)
        return I

    def search(self, data, query):
        self.index.train(data)

        self.index.add(data)

        D, I = self.index.search(query, self.clusters)
        return I
