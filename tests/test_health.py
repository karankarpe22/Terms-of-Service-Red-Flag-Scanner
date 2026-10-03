def test_health_check(client):
    """Verify that the health check endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "tos-red-flag-scanner"
    assert "version" in data


def test_root_endpoint(client):
    """Verify that root endpoint responds with metadata and non-legal disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "disclaimer" in data
    assert "not a legal-advice service" in data["disclaimer"]


def test_placeholder_endpoints_return_501(client):
    """Verify that un-implemented endpoints in Phase 1 return 501 Not Implemented."""
    response = client.post("/api/documents/doc_123/analyze", json={"concern_categories": []})
    assert response.status_code == 501
