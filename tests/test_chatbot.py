import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the NEUChatbot class from app3.py
from app3 import NEUChatbot, DocumentChunk

# Mock the external dependencies
@pytest.fixture
def mock_dependencies(monkeypatch):
    # Mock Pinecone
    mock_pinecone = MagicMock()
    mock_index = MagicMock()
    mock_pinecone.list_indexes.return_value.names.return_value = ["data28k"]
    mock_pinecone.Index.return_value = mock_index
    monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pinecone)
    
    # Mock OpenAI
    mock_openai = MagicMock()
    monkeypatch.setattr("openai.OpenAI", lambda api_key: mock_openai)
    
    # Mock Cohere
    mock_cohere = MagicMock()
    monkeypatch.setattr("cohere.Client", lambda api_key: mock_cohere)
    
    # Mock transformers
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda model_name: mock_tokenizer)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda model_name: mock_model)
    
    return {
        "pinecone": mock_pinecone,
        "index": mock_index,
        "openai": mock_openai,
        "cohere": mock_cohere,
        "tokenizer": mock_tokenizer,
        "model": mock_model
    }

def test_chatbot_initialization(mock_dependencies):
    # Mock streamlit
    with patch("streamlit.sidebar.success"), patch("streamlit.sidebar.info"), patch("streamlit.error"), patch("streamlit.stop"):
        chatbot = NEUChatbot()
        assert chatbot.index is not None
        assert chatbot.cohere_client is not None
        assert chatbot.openai_client is not None
        assert chatbot.tokenizer is not None
        assert chatbot.model is not None

def test_get_embedding(mock_dependencies):
    with patch("streamlit.sidebar.success"), patch("streamlit.sidebar.info"), patch("streamlit.error"), patch("streamlit.stop"):
        # Setup mock return values
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = [0.1, 0.2, 0.3]
        mock_dependencies["model"].return_value = mock_outputs
        
        chatbot = NEUChatbot()
        embedding = chatbot.get_embedding("Test query")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0

def test_document_chunk():
    doc = DocumentChunk("test_id", "This is a test document", 0.95)
    assert doc.pinecone_id == "test_id"
    assert doc.text == "This is a test document"
    assert doc.pinecone_score == 0.95
    assert doc.get_preview(20) == "This is a test docu..."
