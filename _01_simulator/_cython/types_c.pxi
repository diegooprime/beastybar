# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Implementation of C types and state manipulation for BeastyBar.

All functions marked `noexcept nogil` can run without the GIL,
enabling true multi-threaded execution.
"""

from libc.string cimport memcpy, memset
from libc.stdlib cimport rand, srand

# Species lookup tables (initialized at module load time)
# These are module-level constants, no synchronization needed
cdef int8_t SPECIES_STRENGTH[NUM_SPECIES]
cdef int8_t SPECIES_POINTS[NUM_SPECIES]
cdef bint SPECIES_RECURRING[NUM_SPECIES]
cdef bint SPECIES_PERMANENT[NUM_SPECIES]

# Initialize species strengths (indexed by species ID, alphabetically sorted)
# chameleon=5, crocodile=10, giraffe=8, hippo=11, kangaroo=3
# lion=12, monkey=4, parrot=2, seal=6, skunk=1, snake=9, unknown=0, zebra=7
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
# chameleon=3, crocodile=3, giraffe=3, hippo=2, kangaroo=4
# lion=2, monkey=3, parrot=4, seal=2, skunk=4, snake=2, unknown=0, zebra=4
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


cdef void init_species_tables() noexcept nogil:
    """No-op function kept for API compatibility.

    Tables are now initialized at module load time, eliminating the race condition.
    This function does nothing and can be safely called or removed.
    """
    pass


# =============================================================================
# Card helpers
# =============================================================================

cdef Card make_card(int8_t species_id, int8_t owner, int16_t entered_turn) noexcept nogil:
    """Create a new card."""
    cdef Card card
    card.species_id = species_id
    card.owner = owner
    card.entered_turn = entered_turn
    return card


cdef Card empty_card() noexcept nogil:
    """Create an empty card slot marker."""
    cdef Card card
    card.species_id = -1
    card.owner = -1
    card.entered_turn = -1
    return card


cdef bint card_is_empty(Card* card) noexcept nogil:
    """Check if a card slot is empty."""
    return card.species_id < 0


# =============================================================================
# Zone manipulation (for beasty_bar, thats_it)
# =============================================================================

cdef void zone_append(CardZone* zone, Card card) noexcept nogil:
    """Append a card to a zone."""
    if zone.length < TOTAL_CARDS:
        zone.cards[zone.length] = card
        zone.length += 1


cdef Card zone_remove(CardZone* zone, int32_t index) noexcept nogil:
    """Remove and return a card from a zone at index."""
    cdef Card card
    cdef int32_t i

    if index < 0 or index >= zone.length:
        return empty_card()

    card = zone.cards[index]

    # Shift remaining cards left
    for i in range(index, zone.length - 1):
        zone.cards[i] = zone.cards[i + 1]

    zone.length -= 1
    return card


cdef void zone_insert(CardZone* zone, int32_t index, Card card) noexcept nogil:
    """Insert a card at index, shifting others right."""
    cdef int32_t i

    if zone.length >= TOTAL_CARDS:
        return

    if index < 0:
        index = 0
    if index > zone.length:
        index = zone.length

    # Shift cards right
    for i in range(zone.length, index, -1):
        zone.cards[i] = zone.cards[i - 1]

    zone.cards[index] = card
    zone.length += 1


cdef void zone_clear(CardZone* zone) noexcept nogil:
    """Clear all cards from a zone."""
    zone.length = 0


# =============================================================================
# Queue manipulation (max 5 cards)
# =============================================================================

cdef void queue_append(QueueZone* zone, Card card) noexcept nogil:
    """Append a card to the queue."""
    if zone.length < MAX_QUEUE_LENGTH:
        zone.cards[zone.length] = card
        zone.length += 1


cdef Card queue_remove(QueueZone* zone, int32_t index) noexcept nogil:
    """Remove and return a card from the queue at index."""
    cdef Card card
    cdef int32_t i

    if index < 0 or index >= zone.length:
        return empty_card()

    card = zone.cards[index]

    # Shift remaining cards left
    for i in range(index, zone.length - 1):
        zone.cards[i] = zone.cards[i + 1]

    zone.length -= 1
    return card


cdef void queue_insert(QueueZone* zone, int32_t index, Card card) noexcept nogil:
    """Insert a card at index in the queue."""
    cdef int32_t i

    if zone.length >= MAX_QUEUE_LENGTH:
        return

    if index < 0:
        index = 0
    if index > zone.length:
        index = zone.length

    # Shift cards right
    for i in range(zone.length, index, -1):
        zone.cards[i] = zone.cards[i - 1]

    zone.cards[index] = card
    zone.length += 1


cdef void queue_clear(QueueZone* zone) noexcept nogil:
    """Clear all cards from the queue."""
    zone.length = 0


cdef void queue_replace(QueueZone* zone, Card* cards, int32_t length) noexcept nogil:
    """Replace entire queue contents."""
    cdef int32_t i
    cdef int32_t actual_length = length
    if actual_length > MAX_QUEUE_LENGTH:
        actual_length = MAX_QUEUE_LENGTH

    for i in range(actual_length):
        zone.cards[i] = cards[i]
    zone.length = actual_length


# =============================================================================
# Player hand/deck manipulation
# =============================================================================

cdef Card player_remove_hand(PlayerState* player, int32_t index) noexcept nogil:
    """Remove and return a card from player's hand."""
    cdef Card card
    cdef int32_t i

    if index < 0 or index >= player.hand_length:
        return empty_card()

    card = player.hand[index]

    # Shift remaining cards left
    for i in range(index, player.hand_length - 1):
        player.hand[i] = player.hand[i + 1]

    player.hand_length -= 1
    return card


cdef void player_add_hand(PlayerState* player, Card card) noexcept nogil:
    """Add a card to player's hand."""
    if player.hand_length < HAND_SIZE:
        player.hand[player.hand_length] = card
        player.hand_length += 1


cdef Card player_draw_card(PlayerState* player) noexcept nogil:
    """Draw the top card from player's deck to hand."""
    cdef Card card
    cdef int32_t i

    if player.deck_length == 0:
        return empty_card()

    # Take from top of deck (index 0)
    card = player.deck[0]

    # Shift deck left
    for i in range(0, player.deck_length - 1):
        player.deck[i] = player.deck[i + 1]

    player.deck_length -= 1

    # Add to hand
    player_add_hand(player, card)

    return card


# =============================================================================
# State initialization and copying
# =============================================================================

# Simple LCG RNG for nogil shuffling
cdef uint32_t _rng_next(uint32_t* state) noexcept nogil:
    """Linear congruential generator for shuffling."""
    state[0] = state[0] * 1103515245 + 12345
    return (state[0] >> 16) & 0x7fff


cdef void _shuffle_cards(Card* cards, int32_t length, uint32_t* rng_state) noexcept nogil:
    """Fisher-Yates shuffle."""
    cdef int32_t i, j
    cdef Card temp

    for i in range(length - 1, 0, -1):
        j = _rng_next(rng_state) % (i + 1)
        temp = cards[i]
        cards[i] = cards[j]
        cards[j] = temp


cdef void init_game_state(GameState* state, uint32_t seed) noexcept nogil:
    """Initialize a new game state with shuffled decks."""
    cdef int32_t owner, i
    cdef uint32_t rng_state = seed
    cdef Card deck_cards[DECK_SIZE]

    # Clear state
    memset(state, 0, sizeof(GameState))
    state.seed = seed
    state.turn = 0
    state.active_player = 0

    # Initialize each player's deck and hand
    # BASE_DECK order: lion, hippo, crocodile, snake, giraffe, zebra,
    #                  seal, chameleon, monkey, kangaroo, parrot, skunk
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
        # Create deck cards
        for i in range(DECK_SIZE):
            deck_cards[i] = make_card(base_deck_species[i], <int8_t>owner, -1)

        # Shuffle
        _shuffle_cards(deck_cards, DECK_SIZE, &rng_state)

        # First HAND_SIZE cards go to hand
        state.players[owner].hand_length = HAND_SIZE
        for i in range(HAND_SIZE):
            state.players[owner].hand[i] = deck_cards[i]

        # Remaining go to deck
        state.players[owner].deck_length = DECK_SIZE - HAND_SIZE
        for i in range(DECK_SIZE - HAND_SIZE):
            state.players[owner].deck[i] = deck_cards[HAND_SIZE + i]


cdef void copy_game_state(GameState* dest, GameState* src) noexcept nogil:
    """Deep copy a game state."""
    memcpy(dest, src, sizeof(GameState))


cdef int get_active_player_nogil(GameState* state) noexcept nogil:
    """Get active player from state (nogil-safe accessor)."""
    return state.active_player
