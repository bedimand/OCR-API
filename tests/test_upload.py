from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"dummy", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"
