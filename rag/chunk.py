import os
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
from data.data_loader import read_txt_files

# Function to save embeddings to a file using pickle
def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)

# Function to load embeddings from a file if it exists
def load_embeddings(filename="embeddings.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# Function to chunk a long text into smaller segments of a given size
def chunk_text(text, chunk_size=500):
    """
    Splits the text into chunks of the given size.
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to get embeddings for chunks of text.
# If embeddings are already saved, it loads them, otherwise, it computes and saves them.
def get_embeddings_for_chunks(chunks, model, embedding_filename="embeddings.pkl"):
    """
    Get or compute embeddings for chunks of text. If embeddings are already saved, load them.
    Otherwise, compute and save them.
    
    Parameters:
    - chunks: List of text chunks to generate embeddings for.
    - model: Pre-trained model for embeddings (from sentence-transformers).
    - embedding_filename: The filename to load/save embeddings.
    
    Returns:
    - List of embeddings for the chunks.
    """
    # Try to load saved embeddings
    embeddings = load_embeddings(embedding_filename)
    
    if embeddings is None:
        # If no embeddings found, compute and save them
        embeddings = model.encode(chunks, convert_to_tensor=True)
        save_embeddings(embeddings, embedding_filename)  # Save embeddings for future use
    return embeddings

# Function to calculate the top-k most similar chunks to the query using cosine similarity
def get_top_k_similar_chunks(query, text, model, k=3, chunk_size=500, embedding_filename="embeddings.pkl"):
    """
    Uses the RAG technique to split the text, compute similarity, and return top-k similar chunks.
    If embeddings are already saved, load them instead of recalculating.
    
    Parameters:
    - query: The query text.
    - text: The large document/text to be chunked.
    - model: The pre-trained model for embeddings (from sentence-transformers).
    - k: The number of top similar chunks to return.
    - chunk_size: The size of each chunk in the document.
    - embedding_filename: The filename to load/save embeddings.
    
    Returns:
    - List of top-k chunks with highest similarity to the query.
    """
    # Step 1: Chunk the document into smaller parts
    chunks = chunk_text(text, chunk_size)
    
    # Step 2: Get or compute embeddings for the query and the chunks
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = get_embeddings_for_chunks(chunks, model, embedding_filename)
    
    # Step 3: Compute similarity between the query and each chunk
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

    # print("Similarities:", similarities)

    # Check if similarities is empty or contains invalid values
    if similarities.size(0) == 0:
        print("Error: Similarity array is empty.")
        return [], []  # Return empty results or handle the error in another way

    # Convert tensor to NumPy array for easier manipulation
    similarities_np = similarities.cpu().numpy()
    
    # Step 4: Get top-k chunks based on similarity scores
    top_k_indices = np.argsort(similarities_np)[-k:] 
    top_k_chunks = [chunks[idx] for idx in top_k_indices]
    top_k_similarities = similarities_np[top_k_indices]

    return top_k_chunks, top_k_similarities


if __name__ == "__main__":
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read the content from all text files in a directory and merge them into one large text
    directory_path = "data/txt_files"
    all_text = read_txt_files(directory_path)

    # Define the query text
    query_text = "What is the main topic of the document?"

    # Get the top-3 most similar chunks to the query
    top_k_chunks, similarities = get_top_k_similar_chunks(query_text, all_text, model, k=3)

    # Print the results
    for i, chunk in enumerate(top_k_chunks):
        print(f"Rank {i+1}:")
        print(chunk)
        print(f"Similarity: {similarities[i]}")
        print("-" * 50)
