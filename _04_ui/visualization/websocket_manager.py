"""WebSocket connection manager for visualization streaming.

Manages WebSocket connections and broadcasts activation updates
to all connected visualization clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class VisualizerWebSocketManager:
    """Manages WebSocket connections for the neural network visualizer.

    Handles connection lifecycle and broadcasts activation updates
    to all connected clients.
    """

    def __init__(self) -> None:
        """Initialize the WebSocket manager."""
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._session_subscriptions: dict[str, list[WebSocket]] = {}

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept
        """
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info(f"Visualizer client connected. Total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove
        """
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

            # Remove from session subscriptions
            for session_id, sockets in list(self._session_subscriptions.items()):
                if websocket in sockets:
                    sockets.remove(websocket)
                    if not sockets:
                        del self._session_subscriptions[session_id]

        logger.info(f"Visualizer client disconnected. Total: {len(self._connections)}")

    async def subscribe_to_session(
        self, websocket: WebSocket, session_id: str
    ) -> None:
        """Subscribe a connection to a specific game session.

        Args:
            websocket: The WebSocket connection
            session_id: The game session ID to subscribe to
        """
        async with self._lock:
            if session_id not in self._session_subscriptions:
                self._session_subscriptions[session_id] = []
            if websocket not in self._session_subscriptions[session_id]:
                self._session_subscriptions[session_id].append(websocket)

        logger.debug(f"Client subscribed to session {session_id}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: The message dict to broadcast (will be JSON encoded)
        """
        if not self._connections:
            return

        data = json.dumps(message)

        async with self._lock:
            dead_connections: list[WebSocket] = []

            for ws in self._connections:
                try:
                    await ws.send_text(data)
                except Exception as e:
                    logger.debug(f"Failed to send to client: {e}")
                    dead_connections.append(ws)

            # Clean up dead connections
            for ws in dead_connections:
                if ws in self._connections:
                    self._connections.remove(ws)

    async def broadcast_to_session(
        self, session_id: str, message: dict[str, Any]
    ) -> None:
        """Broadcast a message to clients subscribed to a specific session.

        Args:
            session_id: The session ID to broadcast to
            message: The message dict to broadcast
        """
        async with self._lock:
            sockets = self._session_subscriptions.get(session_id, [])

        if not sockets:
            # Fall back to broadcast to all if no specific subscriptions
            await self.broadcast(message)
            return

        data = json.dumps(message)

        async with self._lock:
            dead_connections: list[WebSocket] = []

            for ws in sockets:
                try:
                    await ws.send_text(data)
                except Exception as e:
                    logger.debug(f"Failed to send to client: {e}")
                    dead_connections.append(ws)

            # Clean up dead connections
            for ws in dead_connections:
                if ws in sockets:
                    sockets.remove(ws)
                if ws in self._connections:
                    self._connections.remove(ws)

    async def broadcast_activation(self, activation_dict: dict[str, Any]) -> None:
        """Broadcast an activation update to all clients.

        Args:
            activation_dict: The activation data dict from snapshot_to_dict()
        """
        await self.broadcast(activation_dict)

    async def send_to_client(
        self, websocket: WebSocket, message: dict[str, Any]
    ) -> bool:
        """Send a message to a specific client.

        Args:
            websocket: The target WebSocket connection
            message: The message dict to send

        Returns:
            True if send succeeded, False otherwise
        """
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.debug(f"Failed to send to client: {e}")
            await self.disconnect(websocket)
            return False

    async def handle_client_message(
        self, websocket: WebSocket, data: str
    ) -> dict[str, Any] | None:
        """Handle an incoming message from a client.

        Args:
            websocket: The WebSocket that sent the message
            data: The raw message text

        Returns:
            Response dict to send back, or None for no response
        """
        try:
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "subscribe":
                session_id = message.get("session_id", "default")
                await self.subscribe_to_session(websocket, session_id)
                return {"type": "subscribed", "session_id": session_id}

            elif msg_type == "ping":
                return {"type": "pong"}

            elif msg_type == "request_snapshot":
                # Client is requesting current state - handled by caller
                return {"type": "snapshot_requested"}

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                return {"type": "error", "message": f"Unknown type: {msg_type}"}

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {data[:100]}")
            return {"type": "error", "message": "Invalid JSON"}


# Global instance for the application
visualizer_ws_manager = VisualizerWebSocketManager()
