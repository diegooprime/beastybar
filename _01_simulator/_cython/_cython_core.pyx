# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Main Cython module with OpenMP parallel batch functions.

This module provides the Python interface to the GIL-free game simulation,
enabling multi-threaded execution across all CPU cores.

All internal Cython code is included directly to avoid cross-module cimport issues.
"""

cimport cython
from cython.parallel cimport prange, parallel
from libc.stdlib cimport malloc, free, rand, srand
from libc.stddef cimport size_t
from libc.string cimport memcpy, memset
from libc.stdint cimport int8_t, int16_t, int32_t, uint32_t, uint64_t

import numpy as np
cimport numpy as np

np.import_array()


# =============================================================================
# Inline type declarations (from types_c.pxd)
# =============================================================================

# Game constants (matching rules.py)
cdef enum:
    MAX_QUEUE_LENGTH = 5
    HAND_SIZE = 4
    DECK_SIZE = 12
    PLAYER_COUNT = 2
    TOTAL_CARDS = 24  # DECK_SIZE * PLAYER_COUNT

    # Species IDs (alphabetically sorted to match observations.py _SPECIES_INDEX)
    SPECIES_CHAMELEON = 0
    SPECIES_CROCODILE = 1
    SPECIES_GIRAFFE = 2
    SPECIES_HIPPO = 3
    SPECIES_KANGAROO = 4
    SPECIES_LION = 5
    SPECIES_MONKEY = 6
    SPECIES_PARROT = 7
    SPECIES_SEAL = 8
    SPECIES_SKUNK = 9
    SPECIES_SNAKE = 10
    SPECIES_UNKNOWN = 11
    SPECIES_ZEBRA = 12
    NUM_SPECIES = 13
    NUM_REAL_SPECIES = 12  # Excluding unknown

    # Action space constants
    MAX_PARAM_VALUE = 5  # MAX_QUEUE_LENGTH
    MAX_PARAMS = 2
    ACTIONS_PER_HAND = 31  # 1 + 5 + 5*5 = 1 + 5 + 25 = 31
    ACTION_DIM = 124  # HAND_SIZE * ACTIONS_PER_HAND

    # Observation dimensions (matching observations.py)
    CARD_FEATURE_DIM = 17
    MASKED_CARD_FEATURE_DIM = 3
    SCALARS_DIM = 7
    OBSERVATION_DIM = 988


cdef struct Card:
    int8_t species_id
    int8_t owner
    int16_t entered_turn


cdef struct CardZone:
    Card cards[TOTAL_CARDS]
    int32_t length


cdef struct QueueZone:
    Card cards[MAX_QUEUE_LENGTH]
    int32_t length


cdef struct PlayerState:
    Card hand[HAND_SIZE]
    int32_t hand_length
    Card deck[DECK_SIZE]
    int32_t deck_length


cdef struct GameState:
    QueueZone queue
    CardZone beasty_bar
    CardZone thats_it
    PlayerState players[PLAYER_COUNT]
    int32_t active_player
    int32_t turn
    uint32_t seed


cdef struct Action:
    int8_t hand_index
    int8_t param_count
    int8_t params[MAX_PARAMS]


# Species lookup tables (initialized at module load time)
cdef int8_t SPECIES_STRENGTH[NUM_SPECIES]
cdef int8_t SPECIES_POINTS[NUM_SPECIES]
cdef bint SPECIES_RECURRING[NUM_SPECIES]
cdef bint SPECIES_PERMANENT[NUM_SPECIES]

# Initialize species strengths
SPECIES_STRENGTH[SPECIES_CHAMELEON] = 5
SPECIES_STRENGTH[SPECIES_CROCODILE] = 10
SPECIES_STRENGTH[SPECIES_GIRAFFE] = 8
SPECIES_STRENGTH[SPECIES_HIPPO] = 11
SPECIES_STRENGTH[SPECIES_KANGAROO] = 3
SPECIES_STRENGTH[SPECIES_LION] = 12
SPECIES_STRENGTH[SPECIES_MONKEY] = 4
SPECIES_STRENGTH[SPECIES_PARROT] = 2
SPECIES_STRENGTH[SPECIES_SEAL] = 6
SPECIES_STRENGTH[SPECIES_SKUNK] = 1
SPECIES_STRENGTH[SPECIES_SNAKE] = 9
SPECIES_STRENGTH[SPECIES_UNKNOWN] = 0
SPECIES_STRENGTH[SPECIES_ZEBRA] = 7

# Initialize species points
SPECIES_POINTS[SPECIES_CHAMELEON] = 3
SPECIES_POINTS[SPECIES_CROCODILE] = 3
SPECIES_POINTS[SPECIES_GIRAFFE] = 3
SPECIES_POINTS[SPECIES_HIPPO] = 2
SPECIES_POINTS[SPECIES_KANGAROO] = 4
SPECIES_POINTS[SPECIES_LION] = 2
SPECIES_POINTS[SPECIES_MONKEY] = 3
SPECIES_POINTS[SPECIES_PARROT] = 4
SPECIES_POINTS[SPECIES_SEAL] = 2
SPECIES_POINTS[SPECIES_SKUNK] = 4
SPECIES_POINTS[SPECIES_SNAKE] = 2
SPECIES_POINTS[SPECIES_UNKNOWN] = 0
SPECIES_POINTS[SPECIES_ZEBRA] = 4

# Initialize recurring species (hippo, crocodile, giraffe)
memset(SPECIES_RECURRING, 0, NUM_SPECIES * sizeof(bint))
SPECIES_RECURRING[SPECIES_HIPPO] = True
SPECIES_RECURRING[SPECIES_CROCODILE] = True
SPECIES_RECURRING[SPECIES_GIRAFFE] = True

# Initialize permanent species (zebra)
memset(SPECIES_PERMANENT, 0, NUM_SPECIES * sizeof(bint))
SPECIES_PERMANENT[SPECIES_ZEBRA] = True


# =============================================================================
# Card and zone helper functions
# =============================================================================

cdef inline Card make_card(int8_t species_id, int8_t owner, int16_t entered_turn) noexcept nogil:
    cdef Card card
    card.species_id = species_id
    card.owner = owner
    card.entered_turn = entered_turn
    return card


cdef inline Card empty_card() noexcept nogil:
    cdef Card card
    card.species_id = -1
    card.owner = -1
    card.entered_turn = -1
    return card


cdef inline bint card_is_empty(Card* card) noexcept nogil:
    return card.species_id < 0


cdef inline void zone_append(CardZone* zone, Card card) noexcept nogil:
    if zone.length < TOTAL_CARDS:
        zone.cards[zone.length] = card
        zone.length += 1


cdef inline Card zone_remove(CardZone* zone, int32_t index) noexcept nogil:
    cdef Card card
    cdef int32_t i
    if index < 0 or index >= zone.length:
        return empty_card()
    card = zone.cards[index]
    for i in range(index, zone.length - 1):
        zone.cards[i] = zone.cards[i + 1]
    zone.length -= 1
    return card


cdef inline void zone_insert(CardZone* zone, int32_t index, Card card) noexcept nogil:
    cdef int32_t i
    if zone.length >= TOTAL_CARDS:
        return
    if index < 0:
        index = 0
    if index > zone.length:
        index = zone.length
    for i in range(zone.length, index, -1):
        zone.cards[i] = zone.cards[i - 1]
    zone.cards[index] = card
    zone.length += 1


cdef inline void zone_clear(CardZone* zone) noexcept nogil:
    zone.length = 0


cdef inline void queue_append(QueueZone* zone, Card card) noexcept nogil:
    if zone.length < MAX_QUEUE_LENGTH:
        zone.cards[zone.length] = card
        zone.length += 1


cdef inline Card queue_remove(QueueZone* zone, int32_t index) noexcept nogil:
    cdef Card card
    cdef int32_t i
    if index < 0 or index >= zone.length:
        return empty_card()
    card = zone.cards[index]
    for i in range(index, zone.length - 1):
        zone.cards[i] = zone.cards[i + 1]
    zone.length -= 1
    return card


cdef inline void queue_insert(QueueZone* zone, int32_t index, Card card) noexcept nogil:
    cdef int32_t i
    if zone.length >= MAX_QUEUE_LENGTH:
        return
    if index < 0:
        index = 0
    if index > zone.length:
        index = zone.length
    for i in range(zone.length, index, -1):
        zone.cards[i] = zone.cards[i - 1]
    zone.cards[index] = card
    zone.length += 1


cdef inline void queue_clear(QueueZone* zone) noexcept nogil:
    zone.length = 0


cdef inline void queue_replace(QueueZone* zone, Card* cards, int32_t length) noexcept nogil:
    cdef int32_t i
    cdef int32_t actual_length = length
    if actual_length > MAX_QUEUE_LENGTH:
        actual_length = MAX_QUEUE_LENGTH
    for i in range(actual_length):
        zone.cards[i] = cards[i]
    zone.length = actual_length


cdef inline Card player_remove_hand(PlayerState* player, int32_t index) noexcept nogil:
    cdef Card card
    cdef int32_t i
    if index < 0 or index >= player.hand_length:
        return empty_card()
    card = player.hand[index]
    for i in range(index, player.hand_length - 1):
        player.hand[i] = player.hand[i + 1]
    player.hand_length -= 1
    return card


cdef inline void player_add_hand(PlayerState* player, Card card) noexcept nogil:
    if player.hand_length < HAND_SIZE:
        player.hand[player.hand_length] = card
        player.hand_length += 1


cdef inline Card player_draw_card(PlayerState* player) noexcept nogil:
    cdef Card card
    cdef int32_t i
    if player.deck_length == 0:
        return empty_card()
    card = player.deck[0]
    for i in range(0, player.deck_length - 1):
        player.deck[i] = player.deck[i + 1]
    player.deck_length -= 1
    player_add_hand(player, card)
    return card


# Simple LCG RNG for nogil shuffling
cdef inline uint32_t _rng_next(uint32_t* state) noexcept nogil:
    state[0] = state[0] * 1103515245 + 12345
    return (state[0] >> 16) & 0x7fff


cdef inline void _shuffle_cards(Card* cards, int32_t length, uint32_t* rng_state) noexcept nogil:
    cdef int32_t i, j
    cdef Card temp
    for i in range(length - 1, 0, -1):
        j = _rng_next(rng_state) % (i + 1)
        temp = cards[i]
        cards[i] = cards[j]
        cards[j] = temp


cdef void init_game_state(GameState* state, uint32_t seed) noexcept nogil:
    cdef int32_t owner, i
    cdef uint32_t rng_state = seed
    cdef Card deck_cards[DECK_SIZE]

    # Clear state
    memset(state, 0, sizeof(GameState))
    state.seed = seed
    state.turn = 0
    state.active_player = 0

    # BASE_DECK order
    cdef int8_t base_deck_species[DECK_SIZE]
    base_deck_species[0] = SPECIES_LION
    base_deck_species[1] = SPECIES_HIPPO
    base_deck_species[2] = SPECIES_CROCODILE
    base_deck_species[3] = SPECIES_SNAKE
    base_deck_species[4] = SPECIES_GIRAFFE
    base_deck_species[5] = SPECIES_ZEBRA
    base_deck_species[6] = SPECIES_SEAL
    base_deck_species[7] = SPECIES_CHAMELEON
    base_deck_species[8] = SPECIES_MONKEY
    base_deck_species[9] = SPECIES_KANGAROO
    base_deck_species[10] = SPECIES_PARROT
    base_deck_species[11] = SPECIES_SKUNK

    for owner in range(PLAYER_COUNT):
        for i in range(DECK_SIZE):
            deck_cards[i] = make_card(base_deck_species[i], <int8_t>owner, -1)
        _shuffle_cards(deck_cards, DECK_SIZE, &rng_state)

        state.players[owner].hand_length = HAND_SIZE
        for i in range(HAND_SIZE):
            state.players[owner].hand[i] = deck_cards[i]

        state.players[owner].deck_length = DECK_SIZE - HAND_SIZE
        for i in range(DECK_SIZE - HAND_SIZE):
            state.players[owner].deck[i] = deck_cards[HAND_SIZE + i]


cdef inline void copy_game_state(GameState* dest, GameState* src) noexcept nogil:
    memcpy(dest, src, sizeof(GameState))


# =============================================================================
# Include engine and observation code
# =============================================================================

include "engine_c.pxi"
include "observations_c.pxi"


# =============================================================================
# Python exports
# =============================================================================

# Re-export constants for Python access
PY_SPECIES_CHAMELEON = <int>SPECIES_CHAMELEON
PY_SPECIES_CROCODILE = <int>SPECIES_CROCODILE
PY_SPECIES_GIRAFFE = <int>SPECIES_GIRAFFE
PY_SPECIES_HIPPO = <int>SPECIES_HIPPO
PY_SPECIES_KANGAROO = <int>SPECIES_KANGAROO
PY_SPECIES_LION = <int>SPECIES_LION
PY_SPECIES_MONKEY = <int>SPECIES_MONKEY
PY_SPECIES_PARROT = <int>SPECIES_PARROT
PY_SPECIES_SEAL = <int>SPECIES_SEAL
PY_SPECIES_SKUNK = <int>SPECIES_SKUNK
PY_SPECIES_SNAKE = <int>SPECIES_SNAKE
PY_SPECIES_UNKNOWN = <int>SPECIES_UNKNOWN
PY_SPECIES_ZEBRA = <int>SPECIES_ZEBRA
PY_OBSERVATION_DIM = <int>OBSERVATION_DIM
PY_ACTION_DIM = <int>ACTION_DIM


# =============================================================================
# GameState array wrapper for Python access
# =============================================================================

cdef class GameStateArray:
    """Array of C GameState structs for vectorized operations."""
    cdef GameState* states
    cdef int capacity
    cdef readonly int length

    def __cinit__(self, int capacity):
        self.states = <GameState*>malloc(capacity * sizeof(GameState))
        if self.states == NULL:
            raise MemoryError("Failed to allocate GameStateArray")
        self.capacity = capacity
        self.length = 0

    def __dealloc__(self):
        if self.states != NULL:
            free(self.states)
            self.states = NULL

    def __len__(self):
        return self.length

    cpdef void resize(self, int new_length):
        if new_length > self.capacity or new_length < 0:
            raise ValueError(f"Length {new_length} exceeds capacity {self.capacity}")
        self.length = new_length

    cpdef void init_game(self, int index, unsigned int seed):
        if index < 0 or index >= self.capacity:
            raise IndexError(f"Index {index} out of range")
        init_game_state(&self.states[index], seed)
        if index >= self.length:
            self.length = index + 1

    cpdef bint is_terminal(self, int index):
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range")
        return is_terminal_nogil(&self.states[index])

    cpdef int get_active_player(self, int index):
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range")
        return self.states[index].active_player

    cpdef tuple get_scores(self, int index):
        cdef int scores[2]
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range")
        score_nogil(&self.states[index], scores)
        return (scores[0], scores[1])


# =============================================================================
# Conversion functions: Python State <-> C GameState
# =============================================================================

def python_state_to_c(state, GameStateArray arr, int index):
    """Convert a Python State object to C GameState."""
    cdef GameState* gs = &arr.states[index]
    cdef int i, owner
    cdef int8_t species_id

    species_map = {
        'chameleon': PY_SPECIES_CHAMELEON,
        'crocodile': PY_SPECIES_CROCODILE,
        'giraffe': PY_SPECIES_GIRAFFE,
        'hippo': PY_SPECIES_HIPPO,
        'kangaroo': PY_SPECIES_KANGAROO,
        'lion': PY_SPECIES_LION,
        'monkey': PY_SPECIES_MONKEY,
        'parrot': PY_SPECIES_PARROT,
        'seal': PY_SPECIES_SEAL,
        'skunk': PY_SPECIES_SKUNK,
        'snake': PY_SPECIES_SNAKE,
        'unknown': PY_SPECIES_UNKNOWN,
        'zebra': PY_SPECIES_ZEBRA,
    }

    memset(gs, 0, sizeof(GameState))
    gs.seed = state.seed
    gs.turn = state.turn
    gs.active_player = state.active_player

    gs.queue.length = len(state.zones.queue)
    for i, card in enumerate(state.zones.queue):
        gs.queue.cards[i].species_id = species_map.get(card.species, PY_SPECIES_UNKNOWN)
        gs.queue.cards[i].owner = card.owner
        gs.queue.cards[i].entered_turn = card.entered_turn

    gs.beasty_bar.length = len(state.zones.beasty_bar)
    for i, card in enumerate(state.zones.beasty_bar):
        gs.beasty_bar.cards[i].species_id = species_map.get(card.species, PY_SPECIES_UNKNOWN)
        gs.beasty_bar.cards[i].owner = card.owner
        gs.beasty_bar.cards[i].entered_turn = card.entered_turn

    gs.thats_it.length = len(state.zones.thats_it)
    for i, card in enumerate(state.zones.thats_it):
        gs.thats_it.cards[i].species_id = species_map.get(card.species, PY_SPECIES_UNKNOWN)
        gs.thats_it.cards[i].owner = card.owner
        gs.thats_it.cards[i].entered_turn = card.entered_turn

    for owner in range(2):
        player = state.players[owner]
        gs.players[owner].hand_length = len(player.hand)
        for i, card in enumerate(player.hand):
            gs.players[owner].hand[i].species_id = species_map.get(card.species, PY_SPECIES_UNKNOWN)
            gs.players[owner].hand[i].owner = card.owner
            gs.players[owner].hand[i].entered_turn = card.entered_turn

        gs.players[owner].deck_length = len(player.deck)
        for i, card in enumerate(player.deck):
            gs.players[owner].deck[i].species_id = species_map.get(card.species, PY_SPECIES_UNKNOWN)
            gs.players[owner].deck[i].owner = card.owner
            gs.players[owner].deck[i].entered_turn = card.entered_turn

    if index >= arr.length:
        arr.length = index + 1


# =============================================================================
# Batch functions with OpenMP
# =============================================================================

def step_batch_parallel(
    GameStateArray states,
    np.ndarray[np.int64_t, ndim=1] active_indices,
    np.ndarray[np.int64_t, ndim=1] actions,
    int num_threads = 0
):
    """Step multiple games in parallel using OpenMP."""
    cdef int n = active_indices.shape[0]
    cdef int finished = 0
    cdef int i, idx
    cdef np.int64_t action_idx
    cdef np.int64_t* indices_ptr = &active_indices[0]
    cdef np.int64_t* actions_ptr = &actions[0]
    cdef GameState* states_ptr = states.states
    cdef size_t state_size = sizeof(GameState)

    cdef Action* actions_arr = <Action*>malloc(n * sizeof(Action))
    cdef GameState* new_states = <GameState*>malloc(n * state_size)

    if actions_arr == NULL or new_states == NULL:
        if actions_arr != NULL:
            free(actions_arr)
        if new_states != NULL:
            free(new_states)
        raise MemoryError("Failed to allocate thread-local arrays")

    if num_threads <= 0:
        import os
        num_threads = os.cpu_count() or 4

    with nogil, parallel(num_threads=num_threads):
        for i in prange(n, schedule='dynamic'):
            idx = <int>indices_ptr[i]
            action_idx = actions_ptr[i]
            index_to_action_nogil(<int>action_idx, &actions_arr[i])
            step_nogil(&states_ptr[idx], &actions_arr[i], &new_states[i])
            memcpy(&states_ptr[idx], &new_states[i], state_size)

    free(actions_arr)
    free(new_states)

    for i in range(n):
        idx = <int>indices_ptr[i]
        if is_terminal_nogil(&states_ptr[idx]):
            finished += 1

    return finished


def encode_observations_parallel(
    GameStateArray states,
    np.ndarray[np.int64_t, ndim=1] active_indices,
    np.ndarray[np.float32_t, ndim=2] output,
    int num_threads = 0
):
    """Encode observations for multiple games in parallel."""
    cdef int n = active_indices.shape[0]
    cdef int i, idx, player
    cdef np.int64_t* indices_ptr = &active_indices[0]
    cdef float* output_ptr = &output[0, 0]
    cdef GameState* states_ptr = states.states
    cdef int obs_dim = OBSERVATION_DIM

    if num_threads <= 0:
        import os
        num_threads = os.cpu_count() or 4

    with nogil, parallel(num_threads=num_threads):
        for i in prange(n, schedule='dynamic'):
            idx = <int>indices_ptr[i]
            player = states_ptr[idx].active_player
            state_to_tensor_nogil(&states_ptr[idx], player, &output_ptr[i * obs_dim])


def get_legal_masks_parallel(
    GameStateArray states,
    np.ndarray[np.int64_t, ndim=1] active_indices,
    np.ndarray[np.float32_t, ndim=2] output,
    int num_threads = 0
):
    """Generate legal action masks for multiple games in parallel."""
    cdef int n = active_indices.shape[0]
    cdef int i, idx, player
    cdef np.int64_t* indices_ptr = &active_indices[0]
    cdef float* output_ptr = &output[0, 0]
    cdef GameState* states_ptr = states.states
    cdef int action_dim = ACTION_DIM

    if num_threads <= 0:
        import os
        num_threads = os.cpu_count() or 4

    with nogil, parallel(num_threads=num_threads):
        for i in prange(n, schedule='dynamic'):
            idx = <int>indices_ptr[i]
            player = states_ptr[idx].active_player
            legal_action_mask_nogil(&states_ptr[idx], player, &output_ptr[i * action_dim])


def is_terminal_batch(
    GameStateArray states,
    np.ndarray[np.int64_t, ndim=1] indices
) -> np.ndarray:
    """Check terminal status for multiple games."""
    cdef int n = indices.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] result = np.zeros(n, dtype=np.uint8)
    cdef int i, idx
    cdef np.int64_t* indices_ptr = &indices[0]
    cdef unsigned char* result_ptr = &result[0]
    cdef GameState* states_ptr = states.states

    with nogil:
        for i in range(n):
            idx = <int>indices_ptr[i]
            result_ptr[i] = 1 if is_terminal_nogil(&states_ptr[idx]) else 0

    return result.astype(bool)


# =============================================================================
# Single-game functions for testing
# =============================================================================

def encode_single_observation(GameStateArray states, int index, int perspective = -1):
    """Encode a single observation."""
    cdef np.ndarray[np.float32_t, ndim=1] output = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    cdef int player

    if index < 0 or index >= states.length:
        raise IndexError(f"Index {index} out of range")

    if perspective < 0:
        player = states.states[index].active_player
    else:
        player = perspective

    with nogil:
        state_to_tensor_nogil(&states.states[index], player, &output[0])

    return output


def get_single_legal_mask(GameStateArray states, int index, int perspective = -1):
    """Get legal action mask for a single game."""
    cdef np.ndarray[np.float32_t, ndim=1] output = np.zeros(ACTION_DIM, dtype=np.float32)
    cdef int player

    if index < 0 or index >= states.length:
        raise IndexError(f"Index {index} out of range")

    if perspective < 0:
        player = states.states[index].active_player
    else:
        player = perspective

    with nogil:
        legal_action_mask_nogil(&states.states[index], player, &output[0])

    return output


def step_single(GameStateArray states, int index, int action_index):
    """Apply a single action to a game."""
    cdef Action action
    cdef GameState new_state
    cdef GameState* state_ptr
    cdef size_t state_size = sizeof(GameState)

    if index < 0 or index >= states.length:
        raise IndexError(f"Index {index} out of range")

    state_ptr = &states.states[index]

    with nogil:
        index_to_action_nogil(action_index, &action)
        step_nogil(state_ptr, &action, &new_state)
        memcpy(state_ptr, &new_state, state_size)


def get_single_scores(GameStateArray states, int index):
    """Get scores for a single game."""
    cdef int scores[2]

    if index < 0 or index >= states.length:
        raise IndexError(f"Index {index} out of range")

    score_nogil(&states.states[index], scores)
    return (scores[0], scores[1])


# =============================================================================
# Random action selection (GIL-free for opponent pool)
# =============================================================================

cdef int select_random_action_nogil(GameState* state, int player, uint32_t* rng) noexcept nogil:
    """Select random legal action in pure C for opponent pool.

    Args:
        state: Pointer to the game state
        player: Player index (0 or 1)
        rng: Pointer to RNG state (will be mutated)

    Returns:
        Action index (0-123), or -1 if no legal actions available
    """
    cdef Action legal_actions[ACTION_DIM]
    cdef int count, selected_idx, action_idx

    # Get all legal actions for the player
    count = legal_actions_nogil(state, player, legal_actions)

    if count == 0:
        return -1

    # Select a random action using the provided RNG
    selected_idx = _rng_next(rng) % count

    # Convert the selected Action to an index
    action_idx = action_to_index(&legal_actions[selected_idx])

    return action_idx


def select_random_action(GameStateArray states, int index, unsigned int seed):
    """Python wrapper to select a random legal action.

    Args:
        states: GameStateArray containing the game state
        index: Index of the state in the array
        seed: Random seed for action selection

    Returns:
        Action index (0-123), or -1 if no legal actions available
    """
    cdef uint32_t rng_state = seed
    cdef int player
    cdef int action_idx

    if index < 0 or index >= states.length:
        raise IndexError(f"Index {index} out of range")

    player = states.states[index].active_player

    with nogil:
        action_idx = select_random_action_nogil(&states.states[index], player, &rng_state)

    return action_idx


# =============================================================================
# C GameState to Python State conversion
# =============================================================================

# Species ID to name mapping (reverse of species_map in python_state_to_c)
cdef list _SPECIES_ID_TO_NAME = [
    'chameleon',   # 0 = SPECIES_CHAMELEON
    'crocodile',   # 1 = SPECIES_CROCODILE
    'giraffe',     # 2 = SPECIES_GIRAFFE
    'hippo',       # 3 = SPECIES_HIPPO
    'kangaroo',    # 4 = SPECIES_KANGAROO
    'lion',        # 5 = SPECIES_LION
    'monkey',      # 6 = SPECIES_MONKEY
    'parrot',      # 7 = SPECIES_PARROT
    'seal',        # 8 = SPECIES_SEAL
    'skunk',       # 9 = SPECIES_SKUNK
    'snake',       # 10 = SPECIES_SNAKE
    'unknown',     # 11 = SPECIES_UNKNOWN
    'zebra',       # 12 = SPECIES_ZEBRA
]


def c_state_to_python(GameStateArray arr, int index):
    """Convert C GameState at index to Python State for heuristic agents.

    Args:
        arr: GameStateArray containing the C state
        index: Index of the state in the array

    Returns:
        Python State object with all fields populated
    """
    from _01_simulator import state as state_module

    cdef GameState* gs
    cdef int i, owner
    cdef Card c_card

    if index < 0 or index >= arr.length:
        raise IndexError(f"Index {index} out of range")

    gs = &arr.states[index]

    # Convert queue cards
    queue_cards = []
    for i in range(gs.queue.length):
        c_card = gs.queue.cards[i]
        queue_cards.append(state_module.Card(
            owner=c_card.owner,
            species=_SPECIES_ID_TO_NAME[c_card.species_id],
            entered_turn=c_card.entered_turn
        ))

    # Convert beasty_bar cards
    beasty_bar_cards = []
    for i in range(gs.beasty_bar.length):
        c_card = gs.beasty_bar.cards[i]
        beasty_bar_cards.append(state_module.Card(
            owner=c_card.owner,
            species=_SPECIES_ID_TO_NAME[c_card.species_id],
            entered_turn=c_card.entered_turn
        ))

    # Convert thats_it cards
    thats_it_cards = []
    for i in range(gs.thats_it.length):
        c_card = gs.thats_it.cards[i]
        thats_it_cards.append(state_module.Card(
            owner=c_card.owner,
            species=_SPECIES_ID_TO_NAME[c_card.species_id],
            entered_turn=c_card.entered_turn
        ))

    # Create zones
    zones = state_module.Zones(
        queue=tuple(queue_cards),
        beasty_bar=tuple(beasty_bar_cards),
        thats_it=tuple(thats_it_cards)
    )

    # Convert player states
    player_states = []
    for owner in range(PLAYER_COUNT):
        # Convert hand cards
        hand_cards = []
        for i in range(gs.players[owner].hand_length):
            c_card = gs.players[owner].hand[i]
            hand_cards.append(state_module.Card(
                owner=c_card.owner,
                species=_SPECIES_ID_TO_NAME[c_card.species_id],
                entered_turn=c_card.entered_turn
            ))

        # Convert deck cards
        deck_cards = []
        for i in range(gs.players[owner].deck_length):
            c_card = gs.players[owner].deck[i]
            deck_cards.append(state_module.Card(
                owner=c_card.owner,
                species=_SPECIES_ID_TO_NAME[c_card.species_id],
                entered_turn=c_card.entered_turn
            ))

        player_states.append(state_module.PlayerState(
            deck=tuple(deck_cards),
            hand=tuple(hand_cards)
        ))

    # Create and return the State object
    return state_module.State(
        seed=gs.seed,
        turn=gs.turn,
        active_player=gs.active_player,
        players=tuple(player_states),
        zones=zones
    )
