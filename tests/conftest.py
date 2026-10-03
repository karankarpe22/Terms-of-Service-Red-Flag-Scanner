import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """FastAPI synchronous test client fixture."""
    with TestClient(app) as test_client:
        yield test_client
