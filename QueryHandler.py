class QueryHandler:
    def __init__(self, index, model, chunks):
        self.index = index
        self.model = model
        self.chunks = chunks

    def handle_query(self, query):
        query_embedding = self.model.encode(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(query_embedding, k=5)  # Retrieve top 5 relevant chunks
        return [self.chunks[i] for i in I[0]]
