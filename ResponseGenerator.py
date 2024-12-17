class ResponseGenerator:  
    def __init__(self, model):
        # Load a pre-trained language model for text generation
        self.model = model
        self.max_length = 1024  # Set the maximum length for the model

    def generate_response(self, context, query):  
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        # Truncate the prompt if it exceeds the maximum length
        if len(prompt) > self.max_length:
            prompt = prompt[:self.max_length]    

        # Use max_new_tokens to specify how many new tokens to generate
        response = self.model(prompt, max_new_tokens=50, num_return_sequences=1)  # Generate 50 new tokens
        return response[0]['generated_text']


if __name__ == "__main__":
    urls = ["http://www.vmtw.in"]  # List of URLs to scrape
    data_ingestion = DataIngestion(urls)
    data_ingestion.scrape()
    data_ingestion.chunk_and_embed()
    data_ingestion.store_embeddings()

    query_handler = QueryHandler(data_ingestion.index, data_ingestion.model, data_ingestion.chunks)

    # Example user query
    user_query = input("Enter any query :")
    relevant_chunks = query_handler.handle_query(user_query)

    # Generate response using a language model
    gpt2_model = pipeline("text-generation", model="gpt2")  # Load the model
    response_generator = ResponseGenerator(gpt2_model)  # Pass the model to the ResponseGenerator
    response = response_generator.generate_response(" ".join(relevant_chunks), user_query)

    print("Response:", response)
