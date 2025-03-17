import streamlit as st
import pinecone
from openai import OpenAI
import cohere
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.memory import ConversationBufferMemory
import pandas as pd
import json
import time
import warnings
import uuid
import logging
import transformers
import os

# Get API keys from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")
transformers.logging.set_verbosity_error()

# Set page configuration
st.set_page_config(
    page_title="Northeastern University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []

if 'reranked_chunks' not in st.session_state:
    st.session_state.reranked_chunks = []

# Constants and API keys
INDEX_NAME = "data28k"

class DocumentChunk:
    """Class to represent a document chunk with tracking information"""
    def __init__(self, pinecone_id, text, score):
        self.track_id = str(uuid.uuid4())[:8]  # Unique tracking ID
        self.pinecone_id = pinecone_id
        self.text = text
        self.pinecone_score = score
        self.rerank_score = None
        self.original_rank = None
        self.reranked_rank = None
    
    def get_preview(self, max_length=60):
        """Get a preview of the document text"""
        return self.text[:max_length-3] + "..." if len(self.text) > max_length else self.text

class NEUChatbot:
    def __init__(self):
        """Initialize the NEU Chatbot with necessary clients and models"""
        self.setup_clients()
        self.memory = ConversationBufferMemory()
        self.document_chunks = []  # Store document chunks for tracking
        
    def setup_clients(self):
        """Set up all required API clients"""
        with st.spinner("Initializing services..."):
            # Initialize Pinecone
            self.pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            
            if INDEX_NAME in self.pc.list_indexes().names():
                self.index = self.pc.Index(INDEX_NAME)
                st.sidebar.success(f"Connected to Pinecone index: {INDEX_NAME}")
            else:
                st.error(f"Index '{INDEX_NAME}' does not exist.")
                st.stop()
            
            # Initialize other clients
            self.cohere_client = cohere.Client(api_key=COHERE_API_KEY)
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Load embedding model
            st.sidebar.info("Loading embedding model...")
            self.tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-v1.5")
            self.model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-m-v1.5")
            st.sidebar.success("Embedding model loaded")
    
    def get_embedding(self, text):
        """Generate embeddings for input text"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings.tolist()
    
    def metadata_to_text(self, metadata):
        """Extract text from metadata"""
        if '_node_content' in metadata:
            try:
                content_json = json.loads(metadata['_node_content'])
                return content_json.get('text', '').strip()
            except json.JSONDecodeError:
                return metadata['_node_content'].strip()
        elif 'text' in metadata:
            return metadata['text'].strip()
        else:
            return ''
    
    def query_pinecone(self, query_vector):
        """Query Pinecone for relevant documents"""
        response = self.index.query(vector=query_vector, top_k=10, include_metadata=True)
        
        # Clear previous document chunks
        self.document_chunks = []
        
        for rank, match in enumerate(response['matches'], start=1):
            metadata = match.get('metadata', {})
            text_chunk = self.metadata_to_text(metadata)
            if text_chunk:
                # Use Roman numerals for track_id
                roman_numeral = self.to_roman(rank)
                
                # Create a document chunk object for tracking
                doc = DocumentChunk(
                    pinecone_id=match['id'],
                    text=text_chunk,
                    score=match['score']
                )
                doc.track_id = roman_numeral
                doc.original_rank = rank
                
                self.document_chunks.append(doc)
        
        if not self.document_chunks:
            st.warning("No valid text chunks retrieved from Pinecone.")
            return []
            
        return self.document_chunks
    
    def to_roman(self, num):
        """Convert integer to Roman numeral"""
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syms = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = ''
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman_num += syms[i]
                num -= val[i]
            i += 1
        return roman_num
    
    def rerank_documents(self, query, documents):
        """Rerank documents using Cohere's reranking API"""
        if not documents:
            st.warning("No documents to rerank.")
            return []
        
        # Extract text from document objects
        docs_for_reranking = [doc.text for doc in documents]
        
        # Rerank using Cohere
        rerank_response = self.cohere_client.rerank(
            model='rerank-english-v2.0',
            query=query,
            documents=docs_for_reranking,
            top_n=len(docs_for_reranking)
        )
        
        # Create new ordered list based on reranking
        reordered_docs = []
        
        # Update documents with new ranking information
        for new_rank, result in enumerate(rerank_response.results, start=1):
            doc = documents[result.index]
            doc.rerank_score = result.relevance_score
            doc.reranked_rank = new_rank
            reordered_docs.append(doc)
        
        return reordered_docs
    
    def generate_answer(self, context, question):
        """Generate an answer using OpenAI's API"""
        # Join text from context documents
        context_text = "\n\n".join([doc.text for doc in context])
        
        prompt = f"""
        CONTEXT:
        {context_text}
        
        QUESTION:
        {question}
        
        You are a knowledgeable assistant specializing in Northeastern University (NEU) information. Your purpose is to provide friendly, straightforward responses that sound natural and conversational.
        
        Guidelines:
        - Present information about NEU as factual knowledge without mentioning "context," "provided information," or any references to your information sources in the main body of your answer
        - Extract source URLs found at the beginning of each information chunk
        - After your complete answer, include a "Sources" section that lists all relevant URLs
        - If you lack sufficient information to answer fully, simply state: "I don't have enough information about this topic. Please contact Northeastern University directly for more details .  
        - Never fabricate information
        - Maintain professional language regardless of user input
        - If faced with inappropriate language, respond professionally while addressing the underlying question if legitimate
        
        Every information chunk begins with a URL. Include these URLs only in your "Sources" section at the end of your response, never in the main answer.
        This is important, so if you answer from any document make sure you put sources at the end of answer, never miss it. 
        
        Your responses should sound natural and helpful, as if you're simply sharing knowledge about Northeastern University without revealing how you obtained that information.
        """
        
        completion = self.openai_client.chat.completions.create(
            model= "gpt-4o-mini",
            # "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        answer = completion.choices[0].message.content.strip()
        
        # Save to memory
        self.memory.save_context({"input": question}, {"output": answer})
        
        return answer

def create_pinecone_table(document_chunks):
    """Create a DataFrame for Pinecone results"""
    data = []
    for doc in document_chunks:
        data.append({
            "Track ID": doc.track_id,
            "Pinecone ID": doc.pinecone_id[:8] + "...",
            "Score": f"{doc.pinecone_score:.4f}",
            "Content Preview": doc.get_preview(80)
        })
    return pd.DataFrame(data)

def create_reranked_table(reranked_chunks):
    """Create a DataFrame for reranked results"""
    data = []
    for doc in reranked_chunks:
        # Calculate rank change
        rank_change = doc.original_rank - doc.reranked_rank
        
        # Create rank change indicator
        if rank_change > 0:
            rank_indicator = f"↑{rank_change}"
        elif rank_change < 0:
            rank_indicator = f"↓{abs(rank_change)}"
        else:
            rank_indicator = "="
            
        data.append({
            "New Rank": f"{doc.reranked_rank} {rank_indicator}",
            "Track ID": doc.track_id,
            "Original Rank": doc.original_rank,
            "Rerank Score": f"{doc.rerank_score:.4f}",
            "Content Preview": doc.get_preview(80)
        })
    return pd.DataFrame(data)

def main():
    # Create the main header
    st.title("🐶 AskNEU ")
    st.write("Northeastern University Assistant")
    
    # Initialize the chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing the assistant..."):
            st.session_state.chatbot = NEUChatbot()
    
    # Create sidebar for document displays
    st.sidebar.title("Document Retrieval Process")
    
    # Query input
    query = st.chat_input("What would you like to know about Northeastern University?")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Process query when submitted
    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(query)
        
        # Add to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": query})
        
        # Process the query
        with st.spinner("Processing your query..."):
            # Step 1: Generate embeddings
            st.sidebar.subheader("Step 1: Vector Search Results")
            query_vector = st.session_state.chatbot.get_embedding(query)
            
            # Step 2: Query Pinecone
            chunks_from_pinecone = st.session_state.chatbot.query_pinecone(query_vector)
            if chunks_from_pinecone:
                # Display retrieved chunks in sidebar
                pinecone_df = create_pinecone_table(chunks_from_pinecone)
                st.session_state.document_chunks = chunks_from_pinecone
                st.sidebar.dataframe(pinecone_df, use_container_width=True)
                
                # Step 3: Rerank documents
                st.sidebar.subheader("Step 2: Reranked Results")
                reranked_chunks = st.session_state.chatbot.rerank_documents(query, chunks_from_pinecone)
                if reranked_chunks:
                    # Display reranked chunks in sidebar
                    reranked_df = create_reranked_table(reranked_chunks)
                    st.session_state.reranked_chunks = reranked_chunks
                    st.sidebar.dataframe(reranked_df, use_container_width=True)
                    
                    # Select top chunks for context
                    top_chunks = reranked_chunks[:5]
                    
                    # Step 4: Generate answer
                    st.sidebar.subheader("Step 3: Documents Used for Response")
                    used_docs_data = []
                    for doc in top_chunks:
                        used_docs_data.append({
                            "Track ID": doc.track_id,
                            "Reranked Position": doc.reranked_rank,
                            "Original Position": doc.original_rank,
                            "Score": f"{doc.rerank_score:.4f}"
                        })
                    used_docs_df = pd.DataFrame(used_docs_data)
                    st.sidebar.dataframe(used_docs_df, use_container_width=True)
                    
                    # Generate and display the answer
                    with st.chat_message("assistant"):
                        answer = st.session_state.chatbot.generate_answer(top_chunks, query)
                        st.write(answer)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                    
                    # Show movement summary
                    st.sidebar.subheader("Summary Stats")
                    moved_up = sum(1 for doc in reranked_chunks[:5] if doc.original_rank > doc.reranked_rank)
                    moved_down = sum(1 for doc in reranked_chunks[:5] if doc.original_rank < doc.reranked_rank)
                    no_change = sum(1 for doc in reranked_chunks[:5] if doc.original_rank == doc.reranked_rank)
                    
                    st.sidebar.markdown(f"""
                    **Query Processing Summary:**
                    - Retrieved {len(chunks_from_pinecone)} document chunks
                    - Used top 5 most relevant chunks for response
                    - In the top 5 documents: 
                      - {moved_up} moved up in rank
                      - {moved_down} moved down
                      - {no_change} unchanged
                    """)
                else:
                    st.error("Unable to process the documents. Please try a different question.")
            else:
                st.error("Unable to find relevant information. Please try a different question.")

if __name__ == "__main__":
    main()