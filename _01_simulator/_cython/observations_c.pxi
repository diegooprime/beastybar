# GIL-free observation encoding for neural networks.
# This file is included by _cython_core.pyx - no separate imports needed.

# Observation-specific constants
cdef enum:
    MAX_STRENGTH = 12
    MAX_POINTS = 4
    MAX_TURN = 100


cdef void encode_card_features(
    Card* card,
    int position,
    int zone_length,
    int perspective,
    float* output
) noexcept nogil:
    """Encode a single card into CARD_FEATURE_DIM (17) floats.

    Layout:
        [0]:     presence (0 or 1)
        [1]:     owner (0=opponent, 1=self, 0.5=empty)
        [2-13]:  species one-hot (12 dims, excluding 'unknown')
        [14]:    strength normalized [0,1]
        [15]:    points normalized [0,1]
        [16]:    position normalized [0,1]
    """
    cdef int i
    cdef int8_t species_id
    cdef int adjusted_id
    cdef float owner_val
    cdef float strength_norm
    cdef float points_norm
    cdef float position_norm

    # Clear output
    memset(output, 0, CARD_FEATURE_DIM * sizeof(float))

    # Position normalized (always set, even for empty cards)
    if zone_length > 1:
        position_norm = <float>position / <float>(zone_length - 1)
    else:
        position_norm = 0.0
    output[16] = position_norm

    if card_is_empty(card):
        # Empty slot: presence=0, owner=0.5, rest zeros (position already set)
        output[0] = 0.0
        output[1] = 0.5
        return

    # Presence
    output[0] = 1.0

    # Owner encoding
    if card.owner == perspective:
        owner_val = 1.0
    elif card.owner >= 0:
        owner_val = 0.0
    else:
        owner_val = 0.5
    output[1] = owner_val

    # Species one-hot (12 dims, excluding 'unknown')
    species_id = card.species_id
    if species_id >= 0 and species_id != SPECIES_UNKNOWN:
        # Adjust index to skip 'unknown' (which is at index 11)
        adjusted_id = species_id if species_id < SPECIES_UNKNOWN else species_id - 1
        if 0 <= adjusted_id < NUM_REAL_SPECIES:
            output[2 + adjusted_id] = 1.0

    # Strength normalized
    strength_norm = <float>SPECIES_STRENGTH[species_id] / <float>MAX_STRENGTH
    output[14] = strength_norm

    # Points normalized
    points_norm = <float>SPECIES_POINTS[species_id] / <float>MAX_POINTS
    output[15] = points_norm
    # Position already set at start of function


cdef void encode_masked_card_features(
    int present,
    int position,
    int zone_length,
    float* output
) noexcept nogil:
    """Encode opponent hand card with minimal info (MASKED_CARD_FEATURE_DIM=3).

    Layout:
        [0]: presence
        [1]: position normalized
        [2]: padding (0)
    """
    cdef float position_norm

    output[0] = 1.0 if present else 0.0

    if zone_length > 1:
        position_norm = <float>position / <float>(zone_length - 1)
    else:
        position_norm = 0.0
    output[1] = position_norm

    output[2] = 0.0


cdef void state_to_tensor_nogil(
    GameState* state,
    int perspective,
    float* output
) noexcept nogil:
    """Encode full game state to OBSERVATION_DIM (988) tensor.

    Layout (matching observations.py):
        Queue:        5 * 17 = 85
        Beasty Bar:  24 * 17 = 408
        THAT'S IT:   24 * 17 = 408
        Own Hand:     4 * 17 = 68
        Opp Hand:     4 * 3  = 12
        Scalars:             = 7
        Total:              = 988
    """
    cdef int offset = 0
    cdef int i
    cdef int opponent = 1 - perspective
    cdef Card card
    cdef Card empty = empty_card()
    cdef int queue_actual_len
    cdef int bar_actual_len
    cdef int thats_it_actual_len
    cdef int hand_actual_len
    cdef int opp_hand_count
    cdef float* scalars

    # Encode queue (5 * 17 = 85)
    queue_actual_len = state.queue.length
    for i in range(MAX_QUEUE_LENGTH):
        if i < state.queue.length:
            card = state.queue.cards[i]
        else:
            card = empty
        encode_card_features(&card, i, MAX_QUEUE_LENGTH, perspective, &output[offset])
        offset += CARD_FEATURE_DIM

    # Encode Beasty Bar (24 * 17 = 408)
    bar_actual_len = state.beasty_bar.length
    for i in range(TOTAL_CARDS):
        if i < state.beasty_bar.length:
            card = state.beasty_bar.cards[i]
        else:
            card = empty
        encode_card_features(&card, i, TOTAL_CARDS, perspective, &output[offset])
        offset += CARD_FEATURE_DIM

    # Encode THAT'S IT (24 * 17 = 408)
    thats_it_actual_len = state.thats_it.length
    for i in range(TOTAL_CARDS):
        if i < state.thats_it.length:
            card = state.thats_it.cards[i]
        else:
            card = empty
        encode_card_features(&card, i, TOTAL_CARDS, perspective, &output[offset])
        offset += CARD_FEATURE_DIM

    # Encode own hand (4 * 17 = 68)
    hand_actual_len = state.players[perspective].hand_length
    for i in range(HAND_SIZE):
        if i < state.players[perspective].hand_length:
            card = state.players[perspective].hand[i]
        else:
            card = empty
        encode_card_features(&card, i, HAND_SIZE, perspective, &output[offset])
        offset += CARD_FEATURE_DIM

    # Encode opponent hand (masked, 4 * 3 = 12)
    opp_hand_count = state.players[opponent].hand_length
    for i in range(HAND_SIZE):
        encode_masked_card_features(
            1 if i < opp_hand_count else 0,
            i,
            HAND_SIZE,
            &output[offset]
        )
        offset += MASKED_CARD_FEATURE_DIM

    # Encode scalars (7)
    scalars = &output[offset]

    # Deck counts normalized (own, opponent)
    scalars[0] = <float>state.players[perspective].deck_length / <float>DECK_SIZE
    scalars[1] = <float>state.players[opponent].deck_length / <float>DECK_SIZE

    # Hand counts normalized (own, opponent)
    scalars[2] = <float>state.players[perspective].hand_length / <float>HAND_SIZE
    scalars[3] = <float>state.players[opponent].hand_length / <float>HAND_SIZE

    # Is active player
    scalars[4] = 1.0 if state.active_player == perspective else 0.0

    # Turn normalized
    scalars[5] = <float>state.turn / <float>MAX_TURN
    if scalars[5] > 1.0:
        scalars[5] = 1.0

    # Queue length normalized
    scalars[6] = <float>state.queue.length / <float>MAX_QUEUE_LENGTH


# =============================================================================
# Action space encoding
# =============================================================================

cdef int action_to_index(Action* action) noexcept nogil:
    """Convert Action to catalog index using INTERLEAVED layout.

    Matches Python action_space.py catalog structure:
        For each hand_index:
            Index 0: ()
            Index 1: (0,)
            Index 2-6: (0,0), (0,1), (0,2), (0,3), (0,4)
            Index 7: (1,)
            Index 8-12: (1,0), (1,1), (1,2), (1,3), (1,4)
            ...
            Index 25: (4,)
            Index 26-30: (4,0), (4,1), (4,2), (4,3), (4,4)

    Each first_param has a block of 6 entries (1 single + 5 two-param).
    Total per hand: 1 + 5*6 = 31
    Total: 4 * 31 = 124
    """
    cdef int hand_idx = action.hand_index
    cdef int base = hand_idx * ACTIONS_PER_HAND
    cdef int param0, param1

    if action.param_count == 0:
        return base
    elif action.param_count == 1:
        param0 = action.params[0]
        # Interleaved: base + 1 + param0 * 6
        return base + 1 + param0 * 6
    else:  # param_count >= 2
        param0 = action.params[0]
        param1 = action.params[1]
        # Interleaved: base + 2 + param0 * 6 + param1
        return base + 2 + param0 * 6 + param1


cdef void index_to_action_nogil(int index, Action* action) noexcept nogil:
    """Convert catalog index to Action using INTERLEAVED layout.

    Matches Python action_space.py catalog structure where each first_param
    has a block of 6 entries (1 single param + 5 two-param variants).
    """
    cdef int hand_idx = index // ACTIONS_PER_HAND
    cdef int remainder = index % ACTIONS_PER_HAND
    cdef int offset, first_param, position

    action.hand_index = hand_idx

    if remainder == 0:
        # No params
        action.param_count = 0
    else:
        # Determine position within interleaved structure
        offset = remainder - 1
        first_param = offset // 6
        position = offset % 6

        if position == 0:
            # Single param: (first_param,)
            action.param_count = 1
            action.params[0] = first_param
        else:
            # Two params: (first_param, position-1)
            action.param_count = 2
            action.params[0] = first_param
            action.params[1] = position - 1


cdef void legal_action_mask_nogil(
    GameState* state,
    int perspective,
    float* output
) noexcept nogil:
    """Generate legal action mask tensor.

    Output is ACTION_DIM (124) floats with 1.0 for legal, 0.0 for illegal.
    """
    cdef Action legal[ACTION_DIM]
    cdef int count, i, idx

    # Clear output
    memset(output, 0, ACTION_DIM * sizeof(float))

    # Get legal actions
    count = legal_actions_nogil(state, perspective, legal)

    # Set mask bits
    for i in range(count):
        idx = action_to_index(&legal[i])
        if 0 <= idx < ACTION_DIM:
            output[idx] = 1.0
