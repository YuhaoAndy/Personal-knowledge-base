from __future__ import annotations

from typing import Any, Dict, List
import requests


class ApiClient:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def list_documents(self) -> List[Dict[str, Any]]:
        response = requests.get(self._url("/api/documents"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def upload_document(self, file_name: str, file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        files = {
            "file": (file_name, file_bytes, mime_type or "application/octet-stream")
        }
        response = requests.post(self._url("/api/documents/upload"), files=files, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def delete_document(self, file_id: str) -> Dict[str, Any]:
        response = requests.delete(self._url(f"/api/documents/{file_id}"), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def send_message(self, message: str, session_id: str) -> Dict[str, Any]:
        payload = {
            "message": message,
            "session_id": session_id,
        }
        response = requests.post(self._url("/api/chat/send"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_history(self, session_id: str) -> Dict[str, Any]:
        response = requests.get(
            self._url("/api/chat/history"),
            params={"session_id": session_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def clear_history(self, session_id: str) -> Dict[str, Any]:
        response = requests.delete(
            self._url("/api/chat/clear"),
            params={"session_id": session_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_sessions(self) -> List[str]:
        response = requests.get(self._url("/api/chat/sessions"), timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("sessions", [])
