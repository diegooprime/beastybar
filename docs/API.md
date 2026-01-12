# Beasty Bar API Documentation

The Beasty Bar AI provides a REST API for playing games against AI opponents.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Endpoints

### Game Management

#### Create New Game

```http
POST /api/new-game
```

**Request Body:**
```json
{
  "seed": 12345,
  "startingPlayer": 0,
  "humanPlayer": 0,
  "aiOpponent": "neural"
}
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | integer | random | Random seed for reproducibility |
| `startingPlayer` | 0-1 | 0 | Which player goes first |
| `humanPlayer` | 0-1 | 0 | Which player is human |
| `aiOpponent` | string | "heuristic" | AI opponent type |

**Response:**
```json
{
  "seed": 12345,
  "turn": 0,
  "activePlayer": 0,
  "isTerminal": false,
  "humanPlayer": 0,
  "aiOpponent": "neural",
  "isAiTurn": false,
  "queue": [...],
  "zones": {...},
  "hands": [...],
  "legalActions": [...],
  "log": [...]
}
```

---

#### Get Game State

```http
GET /api/state
```

**Response:** Same as `/api/new-game`

---

#### Get Legal Actions

```http
GET /api/legal-actions
```

**Response:**
```json
[
  {
    "handIndex": 0,
    "params": [],
    "card": {"owner": 0, "species": "lion", "strength": 12, "points": 4},
    "label": "Play lion"
  },
  {
    "handIndex": 1,
    "params": [2],
    "card": {"owner": 0, "species": "kangaroo", "strength": 2, "points": 2},
    "label": "Play kangaroo (hop 2)"
  }
]
```

---

#### Apply Action

```http
POST /api/action
```

**Request Body:**
```json
{
  "handIndex": 0,
  "params": []
}
```

**Response:** Updated game state

---

### AI Opponents

#### Get AI Move

```http
POST /api/ai-move
```

Makes the AI opponent take their turn.

**Response:** Updated game state with AI's move applied

---

#### List AI Agents

```http
GET /api/ai-agents
```

**Response:**
```json
[
  {"id": "neural", "name": "PPO iter 949 (Strongest)", "description": "Neural network trained for 949 iterations"},
  {"id": "heuristic", "name": "Heuristic (Default)", "description": "Balanced strategic AI"},
  {"id": "aggressive", "name": "Heuristic (Aggressive)", "description": "High aggression, bar-focused"},
  {"id": "defensive", "name": "Heuristic (Defensive)", "description": "Conservative, low aggression"},
  {"id": "random", "name": "Random", "description": "Plays random legal moves"}
]
```

---

### AI Battle

#### Start AI Battle

```http
POST /api/ai-battle/start
```

Run multiple games between two AI agents.

**Request Body:**
```json
{
  "player1Agent": "neural",
  "player2Agent": "heuristic",
  "numGames": 50
}
```

**Response:**
```json
{
  "player1Agent": "neural",
  "player2Agent": "heuristic",
  "numGames": 50,
  "wins": [35, 12],
  "games": [...]
}
```

---

### Move Explanation (Phase 6)

#### Explain Move

```http
POST /api/explain-move
```

Get an explanation for why the AI chose a particular move.

**Request Body:**
```json
{
  "actionIndex": 1,
  "agentName": "neural"
}
```

**Response:**
```json
{
  "chosen_action": {
    "action_index": 0,
    "hand_index": 0,
    "card_species": "lion",
    "probability": 0.75,
    "rank": 1
  },
  "alternatives": [
    {
      "action_index": 1,
      "hand_index": 1,
      "card_species": "kangaroo",
      "probability": 0.15,
      "rank": 2
    }
  ],
  "confidence": 0.83,
  "value_prediction": 0.25,
  "top_factors": [
    {"feature": "Queue cards", "importance": 0.45, "direction": "positive"},
    {"feature": "Hand cards", "importance": 0.30, "direction": "positive"}
  ],
  "reasoning": "High confidence (83%) in playing lion. Position looks favorable (value: +0.25). Lion: High strength, king of queue."
}
```

---

### Benchmark (Phase 6)

#### Get Benchmark Results

```http
GET /api/benchmark
```

Get inference performance metrics for the neural model.

**Response:**
```json
{
  "model_name": "neural",
  "device": "cuda:0",
  "latency": [
    {"batch_size": 1, "avg_ms": 0.52, "p95_ms": 0.68},
    {"batch_size": 8, "avg_ms": 0.85, "p95_ms": 1.12}
  ],
  "memory_mb": 5.2,
  "throughput_per_sec": 1923
}
```

---

### Visualization

#### WebSocket for Real-time Visualization

```
ws://localhost:8000/ws/visualizer
```

Streams neural network activations in real-time during gameplay.

#### Get Visualization Status

```http
GET /api/viz/status
```

**Response:**
```json
{
  "connected_clients": 2,
  "neural_agents_available": ["neural", "ppo_iter949"]
}
```

---

## Data Types

### Card

```json
{
  "owner": 0,
  "species": "lion",
  "strength": 12,
  "points": 4
}
```

### Species

| Species | Strength | Points | Ability |
|---------|----------|--------|---------|
| lion | 12 | 4 | King of the queue |
| hippo | 11 | 3 | Pushes back cards |
| crocodile | 10 | 3 | Eats cards at front |
| snake | 9 | 3 | Reorders queue |
| giraffe | 8 | 3 | Advances to front |
| zebra | 7 | 2 | Cannot be pushed |
| seal | 6 | 2 | Duplicates abilities |
| kangaroo | 2-5 | 2 | Hops over cards |
| monkey | 4 | 2 | Works in pairs |
| parrot | 3 | 2 | Repeats last action |
| chameleon | 1 | 1 | Copies species |
| skunk | 0 | 1 | Clears the queue |

### Action

```json
{
  "handIndex": 0,
  "params": [2],
  "card": {...},
  "label": "Play kangaroo (hop 2)"
}
```

---

## Error Responses

All errors return JSON with a `detail` field:

```json
{
  "detail": "No active game"
}
```

**Common Status Codes:**
| Code | Description |
|------|-------------|
| 400 | Bad request (invalid action, game over, etc.) |
| 404 | Resource not found (no active game) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Rate Limiting

- **Limit**: 60 requests per minute per IP
- **Header**: `X-RateLimit-Remaining` shows remaining requests

---

## Examples

### Play a Complete Game (curl)

```bash
# Start new game
curl -X POST http://localhost:8000/api/new-game \
  -H "Content-Type: application/json" \
  -d '{"aiOpponent": "neural"}'

# Make your move
curl -X POST http://localhost:8000/api/action \
  -H "Content-Type: application/json" \
  -d '{"handIndex": 0, "params": []}'

# Let AI move
curl -X POST http://localhost:8000/api/ai-move
```

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Start game
resp = requests.post(f"{BASE_URL}/api/new-game", json={"aiOpponent": "neural"})
state = resp.json()

# Game loop
while not state["isTerminal"]:
    if state["isAiTurn"]:
        resp = requests.post(f"{BASE_URL}/api/ai-move")
    else:
        # Pick first legal action
        action = state["legalActions"][0]
        resp = requests.post(f"{BASE_URL}/api/action", json={
            "handIndex": action["handIndex"],
            "params": action["params"]
        })
    state = resp.json()

print(f"Final score: {state['score']}")
```

### JavaScript/Fetch

```javascript
const response = await fetch('/api/new-game', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ aiOpponent: 'neural' })
});
const state = await response.json();
```
