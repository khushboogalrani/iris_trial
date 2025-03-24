import pinecone
from openai import OpenAI
import cohere
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import warnings
import transformers
import config  
# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")
transformers.logging.set_verbosity_error()

class NEUChatbot:
    def __init__(self):
        """Initialize the NEU Chatbot with necessary clients and models from config"""
        self.setup_clients()
        self.setup_langchain()
        
    def setup_clients(self):
        """Set up all required API clients based on configuration"""
        # Initialize Vector Database (Pinecone)
        if config.VECTOR_DB_NAME.lower() == "pinecone":
            self.pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
            
            if config.PINECONE_INDEX_NAME in self.pc.list_indexes().names():
                self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
                print(f"Connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
            else:
                raise ValueError(f"Index '{config.PINECONE_INDEX_NAME}' does not exist.")
        else:
            raise NotImplementedError(f"Vector database '{config.VECTOR_DB_NAME}' is not supported yet.")
        
        # Initialize Reranker client if enabled
        if config.USE_RERANKER:
            if config.RERANKER_PROVIDER.lower() == "cohere":
                self.reranker_client = cohere.Client(api_key=config.COHERE_API_KEY)
                print(f"Cohere reranker initialized with model: {config.RERANKER_MODEL}")
            else:
                raise NotImplementedError(f"Reranker provider '{config.RERANKER_PROVIDER}' is not supported yet.")
        
        # Initialize LLM client
        if config.LLM_PROVIDER.lower() == "openai":
            self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)
            print(f"OpenAI client initialized")
        else:
            raise NotImplementedError(f"LLM provider '{config.LLM_PROVIDER}' is not supported yet.")
        
        # Load embedding model
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
        print("Embedding model loaded")
    
    def setup_langchain(self):
        """Set up LangChain components based on configuration"""
        # Create prompt template from config
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=config.NEU_PROMPT_TEMPLATE
        )
        
        # Set up Chat model based on config
        if config.LLM_PROVIDER.lower() == "openai":
            self.chat_model = ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE, 
                max_tokens=config.LLM_MAX_TOKENS
            )
        else:
            raise NotImplementedError(f"LLM provider '{config.LLM_PROVIDER}' is not supported yet.")
        
        # Create LLM chain
        self.llm_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.prompt_template,
            output_parser=StrOutputParser()
        )
    
    def get_embedding(self, text):
        """Generate embeddings for input text using the configured model"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings.tolist()
    
    def extract_text_from_metadata(self, metadata):
        """Extract text from metadata"""
        import json
        
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
    
    def query_vector_db(self, query_vector, top_k=None):
        """Query vector database for relevant documents based on config"""
        if top_k is None:
            top_k = config.TOP_K_RETRIEVE
            
        if config.VECTOR_DB_NAME.lower() == "pinecone":
            response = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            
            documents = []
            for match in response['matches']:
                metadata = match.get('metadata', {})
                text_chunk = self.extract_text_from_metadata(metadata)
                if text_chunk:
                    documents.append({
                        'id': match['id'],
                        'text': text_chunk,
                        'score': match['score']
                    })
            
            if not documents:
                print("No valid text chunks retrieved from vector database.")
            
            return documents
        else:
            raise NotImplementedError(f"Vector database '{config.VECTOR_DB_NAME}' is not supported yet.")
    
    def rerank_documents(self, query, documents, top_k=None):
        """Rerank documents using the configured reranker if enabled"""
        if not config.USE_RERANKER:
            print("Reranking is disabled in config. Returning original documents.")
            return documents[:config.TOP_K_RERANK] if top_k is None else documents[:top_k]
            
        if not documents:
            print("No documents to rerank.")
            return []
            
        if top_k is None:
            top_k = config.TOP_K_RERANK
        
        if config.RERANKER_PROVIDER.lower() == "cohere":
            # Extract text from document objects
            docs_for_reranking = [doc['text'] for doc in documents]
            
            # Rerank using Cohere
            rerank_response = self.reranker_client.rerank(
                model=config.RERANKER_MODEL,
                query=query,
                documents=docs_for_reranking,
                top_n=top_k
            )
            
            # Create reordered list based on reranking
            reordered_docs = []
            for result in rerank_response.results:
                reordered_docs.append(documents[result.index])
            
            return reordered_docs
        else:
            raise NotImplementedError(f"Reranker provider '{config.RERANKER_PROVIDER}' is not supported yet.")
    
    def generate_answer(self, context_docs, question):
        """Generate an answer using LangChain with the configured LLM"""
        # Join text from context documents
        context_text = "\n\n".join([doc['text'] for doc in context_docs])
        
        # Generate answer using LangChain
        answer = self.llm_chain.invoke({"context": context_text, "question": question})
        
        return answer
    
    def process_query(self, query, top_k_retrieve=None, top_k_rerank=None):
        """Process a query and generate a response using the configured pipeline"""
        if top_k_retrieve is None:
            top_k_retrieve = config.TOP_K_RETRIEVE
            
        if top_k_rerank is None:
            top_k_rerank = config.TOP_K_RERANK
            
        # Step 1: Generate embeddings
        print("Generating embeddings...")
        query_vector = self.get_embedding(query)
        
        # Step 2: Query vector database
        print("Querying vector database...")
        retrieved_docs = self.query_vector_db(query_vector, top_k=top_k_retrieve)
        
        if not retrieved_docs:
            return "I couldn't find any relevant information about that. Please try a different question or contact Northeastern University directly."
        
        # Step 3: Rerank documents if enabled
        if config.USE_RERANKER:
            print("Reranking documents...")
            context_docs = self.rerank_documents(query, retrieved_docs, top_k=top_k_rerank)
        else:
            print("Skipping reranking (disabled in config)...")
            context_docs = retrieved_docs[:top_k_rerank]
        
        if not context_docs:
            return "I encountered an issue while processing your question. Please try again or contact Northeastern University directly."
        
        # Step 4: Generate answer
        print("Generating answer...")
        answer = self.generate_answer(context_docs, query)
        
        return answer

def main():
    # Initialize the chatbot
    print("Initializing NEU Chatbot...")
    chatbot = NEUChatbot()
    
    # Example usage
    query = input("What would you like to know about Northeastern University? ")
    answer = chatbot.process_query(query)
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()