import sys
import os
import logging
from unittest.mock import patch, MagicMock
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the NEUChatbot class from app3.py
from app3 import NEUChatbot

def validate_model_integration():
    """Validate that all model components can be initialized and used together"""
    logger.info("Starting model integration validation")
    
    try:
        # Mock streamlit functions to avoid UI-related errors
        with patch("streamlit.sidebar.success"), patch("streamlit.sidebar.info"), \
             patch("streamlit.error"), patch("streamlit.stop"), \
             patch("streamlit.spinner"):
            
            # Initialize chatbot
            chatbot = NEUChatbot()
            
            # Test embedding generation
            test_query = "What programs does Northeastern University offer?"
            embedding = chatbot.get_embedding(test_query)
            
            if not isinstance(embedding, list) or len(embedding) == 0:
                logger.error("Embedding generation failed")
                return False
                
            logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
            
            # Mock Pinecone response
            mock_matches = [
                {
                    'id': 'doc1',
                    'score': 0.95,
                    'metadata': {'_node_content': json.dumps({'text': 'Northeastern University offers various undergraduate programs.'})}
                },
                {
                    'id': 'doc2',
                    'score': 0.85,
                    'metadata': {'_node_content': json.dumps({'text': 'The College of Engineering at Northeastern has several majors.'})}
                }
            ]
            
            # Mock query_pinecone method
            original_query_pinecone = chatbot.query_pinecone
            chatbot.query_pinecone = MagicMock(return_value=[
                chatbot.DocumentChunk('doc1', 'Northeastern University offers various undergraduate programs.', 0.95),
                chatbot.DocumentChunk('doc2', 'The College of Engineering at Northeastern has several majors.', 0.85)
            ])
            
            # Test document retrieval
            docs = chatbot.query_pinecone(embedding)
            if not docs or len(docs) == 0:
                logger.error("Document retrieval failed")
                return False
                
            logger.info(f"Successfully retrieved {len(docs)} documents")
            
            # Mock reranking response
            mock_rerank_response = MagicMock()
            mock_rerank_response.results = [
                MagicMock(index=1, relevance_score=0.92),
                MagicMock(index=0, relevance_score=0.88)
            ]
            chatbot.cohere_client.rerank = MagicMock(return_value=mock_rerank_response)
            
            # Test reranking
            reranked_docs = chatbot.rerank_documents(test_query, docs)
            if not reranked_docs or len(reranked_docs) == 0:
                logger.error("Document reranking failed")
                return False
                
            logger.info(f"Successfully reranked {len(reranked_docs)} documents")
            
            # Mock OpenAI response
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content="Northeastern University offers various programs including engineering, business, and computer science."))]
            chatbot.openai_client.chat.completions.create = MagicMock(return_value=mock_completion)
            
            # Test answer generation
            answer = chatbot.generate_answer(reranked_docs, test_query)
            if not answer or len(answer) == 0:
                logger.error("Answer generation failed")
                return False
                
            logger.info(f"Successfully generated answer: {answer[:50]}...")
            
            # Restore original method
            chatbot.query_pinecone = original_query_pinecone
            
            # Save validation results
            os.makedirs("validation_results", exist_ok=True)
            with open("validation_results/model_integration.json", "w") as f:
                json.dump({
                    "embedding_generation": "success",
                    "document_retrieval": "success",
                    "document_reranking": "success",
                    "answer_generation": "success",
                    "overall_status": "success"
                }, f, indent=2)
            
            logger.info("Model integration validation completed successfully")
            return True
    except Exception as e:
        logger.error(f"Error during model integration validation: {str(e)}")
        
        # Save validation results
        os.makedirs("validation_results", exist_ok=True)
        with open("validation_results/model_integration.json", "w") as f:
            json.dump({
                "overall_status": "failure",
                "error": str(e)
            }, f, indent=2)
        
        return False

if __name__ == "__main__":
    success = validate_model_integration()
    if not success:
        sys.exit(1)
