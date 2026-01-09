# GIL-free game engine implementation.
# This file is included by _cython_core.pyx - no separate imports needed.


# =============================================================================
# Helper functions
# =============================================================================

cdef int find_card_in_queue(QueueZone* queue, Card* card) noexcept nogil:
    """Find index of card in queue by reference comparison."""
    cdef int i
    for i in range(queue.length):
        if (queue.cards[i].species_id == card.species_id and
            queue.cards[i].owner == card.owner and
            queue.cards[i].entered_turn == card.entered_turn):
            return i
    return -1


cdef int count_species_in_queue(QueueZone* queue, int8_t species_id) noexcept nogil:
    """Count cards of a specific species in the queue."""
    cdef int count = 0
    cdef int i
    for i in range(queue.length):
        if queue.cards[i].species_id == species_id:
            count += 1
    return count


cdef void sort_queue_by_strength(QueueZone* queue) noexcept nogil:
    """Sort queue by strength descending (insertion sort, stable)."""
    cdef int i, j
    cdef Card key
    cdef int8_t key_strength, current_strength

    # Stable insertion sort for small arrays (max 5 elements)
    # Maintains relative order of cards with equal strength
    for i in range(1, queue.length):
        key = queue.cards[i]
        key_strength = SPECIES_STRENGTH[key.species_id]
        j = i - 1

        # Shift elements to the right while they are weaker than key
        # For descending sort: move elements right if current < key
        # Using <= would break stability, < maintains it
        while j >= 0:
            current_strength = SPECIES_STRENGTH[queue.cards[j].species_id]
            if current_strength >= key_strength:
                break
            queue.cards[j + 1] = queue.cards[j]
            j -= 1

        queue.cards[j + 1] = key


cdef void reverse_queue(QueueZone* queue) noexcept nogil:
    """Reverse the order of cards in the queue."""
    cdef int i
    cdef Card temp
    cdef int half = queue.length // 2

    for i in range(half):
        temp = queue.cards[i]
        queue.cards[i] = queue.cards[queue.length - 1 - i]
        queue.cards[queue.length - 1 - i] = temp


# =============================================================================
# Card abilities
# =============================================================================

cdef void ability_lion(GameState* state, Card* card) noexcept nogil:
    """Lion: Move to front, scare monkeys away. Multiple lions fight."""
    cdef QueueZone* queue = &state.queue
    cdef int lion_count = count_species_in_queue(queue, SPECIES_LION)
    cdef int card_idx = find_card_in_queue(queue, card)
    cdef int i
    cdef Card removed, lion_card
    cdef Card remaining[MAX_QUEUE_LENGTH]
    cdef Card scared[MAX_QUEUE_LENGTH]
    cdef int remaining_count = 0
    cdef int scared_count = 0

    if card_idx < 0:
        return

    if lion_count > 1:
        # Multiple lions: the new one goes to THAT'S IT
        removed = queue_remove(queue, card_idx)
        zone_append(&state.thats_it, removed)
        return

    # Save the lion card
    lion_card = queue.cards[card_idx]

    # Collect remaining and scared cards
    for i in range(queue.length):
        if i == card_idx:
            continue  # Skip the lion
        if queue.cards[i].species_id == SPECIES_MONKEY:
            scared[scared_count] = queue.cards[i]
            scared_count += 1
        else:
            remaining[remaining_count] = queue.cards[i]
            remaining_count += 1

    # Send scared monkeys to THAT'S IT
    for i in range(scared_count):
        zone_append(&state.thats_it, scared[i])

    # Rebuild queue: lion first, then remaining
    queue.length = 0
    queue_append(queue, lion_card)
    for i in range(remaining_count):
        queue_append(queue, remaining[i])


cdef void ability_hippo(GameState* state, Card* card) noexcept nogil:
    """Hippo: Push forward, blocked by zebra or stronger/equal."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = find_card_in_queue(queue, card)
    cdef int target = card_idx
    cdef int8_t card_strength = SPECIES_STRENGTH[card.species_id]
    cdef Card hippo_card, ahead

    if card_idx < 0:
        return

    hippo_card = queue.cards[card_idx]

    # Find target position
    while target > 0:
        ahead = queue.cards[target - 1]
        if ahead.species_id == SPECIES_ZEBRA or SPECIES_STRENGTH[ahead.species_id] >= card_strength:
            break
        target -= 1

    if target == card_idx:
        return  # No movement

    # Remove and reinsert
    queue_remove(queue, card_idx)
    queue_insert(queue, target, hippo_card)


cdef void ability_crocodile(GameState* state, Card* card) noexcept nogil:
    """Crocodile: Eat weaker animals ahead, blocked by zebra or stronger/equal."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = find_card_in_queue(queue, card)
    cdef int scan
    cdef int8_t card_strength = SPECIES_STRENGTH[card.species_id]
    cdef Card ahead

    if card_idx < 0:
        return

    scan = card_idx - 1
    while scan >= 0:
        ahead = queue.cards[scan]
        if ahead.species_id == SPECIES_ZEBRA or SPECIES_STRENGTH[ahead.species_id] >= card_strength:
            break
        # Eat this card
        queue_remove(queue, scan)
        zone_append(&state.thats_it, ahead)
        card_idx -= 1  # Adjust our position
        scan -= 1


cdef void ability_snake(GameState* state, Card* card) noexcept nogil:
    """Snake: Sort queue by strength descending."""
    sort_queue_by_strength(&state.queue)


cdef void ability_giraffe(GameState* state, Card* card) noexcept nogil:
    """Giraffe: Move one position forward if ahead is weaker."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = find_card_in_queue(queue, card)
    cdef Card giraffe_card, ahead
    cdef int8_t card_strength = SPECIES_STRENGTH[card.species_id]

    if card_idx < 0 or card_idx == 0:
        return

    ahead = queue.cards[card_idx - 1]
    if SPECIES_STRENGTH[ahead.species_id] < card_strength:
        giraffe_card = queue.cards[card_idx]
        queue.cards[card_idx] = queue.cards[card_idx - 1]
        queue.cards[card_idx - 1] = giraffe_card


cdef void ability_kangaroo(GameState* state, Card* card, int hop_distance) noexcept nogil:
    """Kangaroo: Hop forward by 1-2 positions."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = find_card_in_queue(queue, card)
    cdef int max_hop
    cdef Card kangaroo_card

    if card_idx < 0 or card_idx == 0:
        return

    max_hop = card_idx if card_idx < 2 else 2
    if hop_distance <= 0:
        hop_distance = max_hop
    if hop_distance > max_hop:
        hop_distance = max_hop

    kangaroo_card = queue_remove(queue, card_idx)
    queue_insert(queue, card_idx - hop_distance, kangaroo_card)


cdef void ability_monkey(GameState* state, Card* card) noexcept nogil:
    """Monkey: If 2+ monkeys, expel hippos and crocodiles, monkeys move to front."""
    cdef QueueZone* queue = &state.queue
    cdef int monkey_count = count_species_in_queue(queue, SPECIES_MONKEY)
    cdef int i
    cdef Card survivors[MAX_QUEUE_LENGTH]
    cdef Card expelled[MAX_QUEUE_LENGTH]
    cdef Card monkeys[MAX_QUEUE_LENGTH]
    cdef Card others[MAX_QUEUE_LENGTH]
    cdef int survivor_count = 0
    cdef int expelled_count = 0
    cdef int monkey_idx = 0
    cdef int other_idx = 0
    cdef int8_t species

    if monkey_count < 2:
        return

    # Separate cards into survivors and expelled
    for i in range(queue.length):
        species = queue.cards[i].species_id
        if species == SPECIES_HIPPO or species == SPECIES_CROCODILE:
            expelled[expelled_count] = queue.cards[i]
            expelled_count += 1
        else:
            survivors[survivor_count] = queue.cards[i]
            survivor_count += 1

    # Send expelled to THAT'S IT
    for i in range(expelled_count):
        zone_append(&state.thats_it, expelled[i])

    # Separate survivors into monkeys and others
    for i in range(survivor_count):
        if survivors[i].species_id == SPECIES_MONKEY:
            monkeys[monkey_idx] = survivors[i]
            monkey_idx += 1
        else:
            others[other_idx] = survivors[i]
            other_idx += 1

    # Rebuild queue: current monkey first, then other monkeys (reversed), then others
    queue.length = 0
    # Find the played card among monkeys
    cdef int played_idx = -1
    for i in range(monkey_idx):
        if (monkeys[i].species_id == card.species_id and
            monkeys[i].owner == card.owner and
            monkeys[i].entered_turn == card.entered_turn):
            played_idx = i
            break

    if played_idx >= 0:
        queue_append(queue, monkeys[played_idx])
        # Add other monkeys in reverse order
        for i in range(monkey_idx - 1, -1, -1):
            if i != played_idx:
                queue_append(queue, monkeys[i])
    else:
        # Fallback: add all monkeys
        for i in range(monkey_idx):
            queue_append(queue, monkeys[i])

    # Add remaining others
    for i in range(other_idx):
        queue_append(queue, others[i])


cdef void ability_parrot(GameState* state, Card* card, int target_index) noexcept nogil:
    """Parrot: Send target card to THAT'S IT."""
    cdef QueueZone* queue = &state.queue
    cdef Card target

    if target_index < 0 or target_index >= queue.length:
        return

    target = queue_remove(queue, target_index)
    zone_append(&state.thats_it, target)


cdef void ability_seal(GameState* state, Card* card) noexcept nogil:
    """Seal: Reverse the queue order."""
    reverse_queue(&state.queue)


cdef void ability_chameleon(GameState* state, Card* card, Action* action) noexcept nogil:
    """Chameleon: Copy another card's ability."""
    cdef QueueZone* queue = &state.queue
    cdef int target_index
    cdef int extra_param = 0
    cdef Card target_card, fake_card
    cdef int card_idx
    cdef int i
    cdef int8_t target_species

    if action.param_count < 1:
        return

    target_index = action.params[0]
    if target_index < 0 or target_index >= queue.length:
        return

    target_card = queue.cards[target_index]
    target_species = target_card.species_id

    # Can't copy another chameleon
    if target_species == SPECIES_CHAMELEON:
        return

    # Get extra param if available
    if action.param_count >= 2:
        extra_param = action.params[1]

    # Temporarily change chameleon to target species in the queue
    card_idx = find_card_in_queue(queue, card)
    if card_idx >= 0:
        fake_card = queue.cards[card_idx]
        fake_card.species_id = target_species
        queue.cards[card_idx] = fake_card

        # Execute the target ability
        if target_species == SPECIES_LION:
            ability_lion(state, &queue.cards[card_idx])
        elif target_species == SPECIES_HIPPO:
            ability_hippo(state, &queue.cards[card_idx])
        elif target_species == SPECIES_CROCODILE:
            ability_crocodile(state, &queue.cards[card_idx])
        elif target_species == SPECIES_SNAKE:
            ability_snake(state, &queue.cards[card_idx])
        elif target_species == SPECIES_GIRAFFE:
            ability_giraffe(state, &queue.cards[card_idx])
        elif target_species == SPECIES_KANGAROO:
            ability_kangaroo(state, &queue.cards[card_idx], extra_param)
        elif target_species == SPECIES_MONKEY:
            ability_monkey(state, &queue.cards[card_idx])
        elif target_species == SPECIES_PARROT:
            ability_parrot(state, &queue.cards[card_idx], extra_param)
        elif target_species == SPECIES_SEAL:
            ability_seal(state, &queue.cards[card_idx])
        elif target_species == SPECIES_SKUNK:
            ability_skunk(state, &queue.cards[card_idx])

        # Change back to chameleon (find it again since queue may have changed)
        for i in range(state.queue.length):
            if (state.queue.cards[i].owner == card.owner and
                state.queue.cards[i].entered_turn == card.entered_turn):
                state.queue.cards[i].species_id = SPECIES_CHAMELEON
                break


cdef void ability_skunk(GameState* state, Card* card) noexcept nogil:
    """Skunk: Expel top 2 strength species."""
    cdef QueueZone* queue = &state.queue
    cdef int i, j
    cdef int8_t species
    cdef int8_t strength
    cdef int8_t top_two[2]
    cdef int top_count = 0
    cdef int8_t seen_species[NUM_SPECIES]
    cdef int8_t species_strengths[NUM_SPECIES]
    cdef int species_count = 0
    cdef Card remaining[MAX_QUEUE_LENGTH]
    cdef int remaining_count = 0
    cdef bint found, expelled
    cdef int max_idx
    cdef int8_t temp_s, temp_str

    memset(seen_species, -1, NUM_SPECIES)

    # Find unique species in queue (excluding skunk)
    for i in range(queue.length):
        species = queue.cards[i].species_id
        if species == SPECIES_SKUNK:
            continue
        # Check if already seen
        found = False
        for j in range(species_count):
            if seen_species[j] == species:
                found = True
                break
        if not found:
            seen_species[species_count] = species
            species_strengths[species_count] = SPECIES_STRENGTH[species]
            species_count += 1

    if species_count == 0:
        return

    # Sort to find top 2 by strength (simple selection)
    for i in range(min(2, species_count)):
        max_idx = i
        for j in range(i + 1, species_count):
            if species_strengths[j] > species_strengths[max_idx]:
                max_idx = j
        # Swap
        temp_s = seen_species[i]
        temp_str = species_strengths[i]
        seen_species[i] = seen_species[max_idx]
        species_strengths[i] = species_strengths[max_idx]
        seen_species[max_idx] = temp_s
        species_strengths[max_idx] = temp_str

    top_count = min(2, species_count)
    for i in range(top_count):
        top_two[i] = seen_species[i]

    # Filter queue
    for i in range(queue.length):
        species = queue.cards[i].species_id
        if species == SPECIES_SKUNK:
            remaining[remaining_count] = queue.cards[i]
            remaining_count += 1
        else:
            # Check if in top two
            expelled = False
            for j in range(top_count):
                if species == top_two[j]:
                    expelled = True
                    break
            if expelled:
                zone_append(&state.thats_it, queue.cards[i])
            else:
                remaining[remaining_count] = queue.cards[i]
                remaining_count += 1

    # Replace queue
    queue_replace(queue, remaining, remaining_count)


# =============================================================================
# Recurring abilities
# =============================================================================

cdef void recurring_hippo(GameState* state, int* index) noexcept nogil:
    """Recurring hippo movement."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = index[0]
    cdef int target = card_idx
    cdef int8_t card_strength
    cdef Card hippo_card, ahead

    if card_idx >= queue.length:
        index[0] += 1
        return

    hippo_card = queue.cards[card_idx]
    card_strength = SPECIES_STRENGTH[hippo_card.species_id]

    while target > 0:
        ahead = queue.cards[target - 1]
        if ahead.species_id == SPECIES_ZEBRA or SPECIES_STRENGTH[ahead.species_id] >= card_strength:
            break
        target -= 1

    if target == card_idx:
        index[0] += 1
        return

    queue_remove(queue, card_idx)
    queue_insert(queue, target, hippo_card)
    index[0] = target + 1


cdef void recurring_crocodile(GameState* state, int* index) noexcept nogil:
    """Recurring crocodile eating."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = index[0]
    cdef int scan
    cdef int8_t card_strength
    cdef Card croc_card, ahead

    if card_idx >= queue.length:
        index[0] += 1
        return

    croc_card = queue.cards[card_idx]
    card_strength = SPECIES_STRENGTH[croc_card.species_id]

    scan = card_idx - 1
    while scan >= 0:
        ahead = queue.cards[scan]
        if ahead.species_id == SPECIES_ZEBRA or SPECIES_STRENGTH[ahead.species_id] >= card_strength:
            break
        queue_remove(queue, scan)
        zone_append(&state.thats_it, ahead)
        card_idx -= 1
        scan -= 1

    index[0] = card_idx + 1


cdef void recurring_giraffe(GameState* state, int* index) noexcept nogil:
    """Recurring giraffe stepping forward."""
    cdef QueueZone* queue = &state.queue
    cdef int card_idx = index[0]
    cdef Card giraffe_card, ahead
    cdef int8_t card_strength

    if card_idx >= queue.length:
        index[0] += 1
        return

    giraffe_card = queue.cards[card_idx]

    # Skip if entered this turn
    if giraffe_card.entered_turn == state.turn:
        index[0] += 1
        return

    if card_idx == 0:
        index[0] += 1
        return

    card_strength = SPECIES_STRENGTH[giraffe_card.species_id]
    ahead = queue.cards[card_idx - 1]

    if SPECIES_STRENGTH[ahead.species_id] >= card_strength:
        index[0] += 1
        return

    # Swap with card ahead
    queue.cards[card_idx] = ahead
    queue.cards[card_idx - 1] = giraffe_card
    # Don't increment index so we check again


cdef void process_recurring_nogil(GameState* state) noexcept nogil:
    """Process all recurring abilities in the queue."""
    cdef int index = 0
    cdef int8_t species

    while index < state.queue.length:
        species = state.queue.cards[index].species_id
        if species == SPECIES_HIPPO:
            recurring_hippo(state, &index)
        elif species == SPECIES_CROCODILE:
            recurring_crocodile(state, &index)
        elif species == SPECIES_GIRAFFE:
            recurring_giraffe(state, &index)
        else:
            index += 1


cdef void apply_five_card_check_nogil(GameState* state) noexcept nogil:
    """Apply the five-card queue check."""
    cdef QueueZone* queue = &state.queue
    cdef Card entering[2]
    cdef Card bounced
    cdef Card remaining[MAX_QUEUE_LENGTH]
    cdef int i

    if queue.length != MAX_QUEUE_LENGTH:
        return

    # First 2 enter Beasty Bar
    entering[0] = queue.cards[0]
    entering[1] = queue.cards[1]

    # Last 1 bounces to THAT'S IT
    bounced = queue.cards[queue.length - 1]

    # Middle cards remain
    for i in range(2, queue.length - 1):
        remaining[i - 2] = queue.cards[i]

    # Apply changes
    queue_replace(queue, remaining, queue.length - 3)
    zone_append(&state.beasty_bar, entering[0])
    zone_append(&state.beasty_bar, entering[1])
    zone_append(&state.thats_it, bounced)


cdef void resolve_play_nogil(GameState* state, Card* played_card, Action* action) noexcept nogil:
    """Apply the played card's ability."""
    cdef int8_t species = played_card.species_id
    cdef int param0 = 0
    cdef int param1 = 0

    if action.param_count >= 1:
        param0 = action.params[0]
    if action.param_count >= 2:
        param1 = action.params[1]

    if species == SPECIES_LION:
        ability_lion(state, played_card)
    elif species == SPECIES_HIPPO:
        ability_hippo(state, played_card)
    elif species == SPECIES_CROCODILE:
        ability_crocodile(state, played_card)
    elif species == SPECIES_SNAKE:
        ability_snake(state, played_card)
    elif species == SPECIES_GIRAFFE:
        ability_giraffe(state, played_card)
    elif species == SPECIES_KANGAROO:
        ability_kangaroo(state, played_card, param0)
    elif species == SPECIES_MONKEY:
        ability_monkey(state, played_card)
    elif species == SPECIES_PARROT:
        ability_parrot(state, played_card, param0)
    elif species == SPECIES_SEAL:
        ability_seal(state, played_card)
    elif species == SPECIES_CHAMELEON:
        ability_chameleon(state, played_card, action)
    elif species == SPECIES_SKUNK:
        ability_skunk(state, played_card)


# =============================================================================
# Core engine functions
# =============================================================================

cdef void step_nogil(GameState* state, Action* action, GameState* result) noexcept nogil:
    """Apply an action to advance the game state."""
    cdef int player
    cdef Card played_card
    cdef int hand_idx

    # Copy state
    copy_game_state(result, state)

    player = result.active_player
    hand_idx = action.hand_index

    # Validate hand index
    if hand_idx < 0 or hand_idx >= result.players[player].hand_length:
        return

    # Remove card from hand
    played_card = player_remove_hand(&result.players[player], hand_idx)
    played_card.entered_turn = result.turn

    # Add to queue
    queue_append(&result.queue, played_card)

    # Resolve card ability
    resolve_play_nogil(result, &played_card, action)

    # Process recurring abilities
    process_recurring_nogil(result)

    # Apply five-card check
    apply_five_card_check_nogil(result)

    # Draw a card for the player
    player_draw_card(&result.players[player])

    # Switch to next player and advance turn
    result.active_player = 1 - player
    result.turn += 1


cdef bint is_terminal_nogil(GameState* state) noexcept nogil:
    """Check if the game has ended."""
    cdef int remaining_cards = 0
    cdef int i

    if state.turn == 0:
        return False

    remaining_cards = state.queue.length
    for i in range(PLAYER_COUNT):
        remaining_cards += state.players[i].hand_length
        remaining_cards += state.players[i].deck_length

    if remaining_cards < MAX_QUEUE_LENGTH:
        return True

    # Check if all players have empty hands and decks
    for i in range(PLAYER_COUNT):
        if state.players[i].hand_length > 0 or state.players[i].deck_length > 0:
            return False

    return True


cdef int legal_actions_nogil(GameState* state, int player, Action* out_actions) noexcept nogil:
    """Generate all legal actions for a player. Returns count."""
    cdef int count = 0
    cdef int hand_idx, hop, target, extra_target
    cdef int8_t species, target_species
    cdef int queue_len = state.queue.length
    cdef int max_hop
    cdef PlayerState* player_state = &state.players[player]
    cdef Card card, target_card

    if player != state.active_player:
        return 0

    for hand_idx in range(player_state.hand_length):
        card = player_state.hand[hand_idx]
        species = card.species_id

        if species == SPECIES_KANGAROO:
            max_hop = queue_len if queue_len < 2 else 2
            if max_hop == 0:
                # No params needed
                out_actions[count].hand_index = hand_idx
                out_actions[count].param_count = 0
                count += 1
            else:
                for hop in range(1, max_hop + 1):
                    out_actions[count].hand_index = hand_idx
                    out_actions[count].param_count = 1
                    out_actions[count].params[0] = hop
                    count += 1

        elif species == SPECIES_PARROT:
            for target in range(queue_len):
                out_actions[count].hand_index = hand_idx
                out_actions[count].param_count = 1
                out_actions[count].params[0] = target
                count += 1

        elif species == SPECIES_CHAMELEON:
            if queue_len == 0:
                continue  # Chameleon needs a target
            for target in range(queue_len):
                target_card = state.queue.cards[target]
                target_species = target_card.species_id
                if target_species == SPECIES_CHAMELEON:
                    continue  # Can't copy chameleon

                if target_species == SPECIES_PARROT:
                    # Chameleon-as-parrot needs extra target
                    for extra_target in range(queue_len):
                        out_actions[count].hand_index = hand_idx
                        out_actions[count].param_count = 2
                        out_actions[count].params[0] = target
                        out_actions[count].params[1] = extra_target
                        count += 1
                elif target_species == SPECIES_KANGAROO:
                    # Chameleon-as-kangaroo
                    max_hop = queue_len if queue_len < 2 else 2
                    if max_hop == 0:
                        out_actions[count].hand_index = hand_idx
                        out_actions[count].param_count = 1
                        out_actions[count].params[0] = target
                        count += 1
                    else:
                        for hop in range(1, max_hop + 1):
                            out_actions[count].hand_index = hand_idx
                            out_actions[count].param_count = 2
                            out_actions[count].params[0] = target
                            out_actions[count].params[1] = hop
                            count += 1
                else:
                    out_actions[count].hand_index = hand_idx
                    out_actions[count].param_count = 1
                    out_actions[count].params[0] = target
                    count += 1
        else:
            # No params needed
            out_actions[count].hand_index = hand_idx
            out_actions[count].param_count = 0
            count += 1

    return count


cdef void score_nogil(GameState* state, int* scores) noexcept nogil:
    """Calculate final scores."""
    cdef int i
    cdef Card card

    scores[0] = 0
    scores[1] = 0

    for i in range(state.beasty_bar.length):
        card = state.beasty_bar.cards[i]
        if card.owner >= 0 and card.owner < PLAYER_COUNT:
            scores[card.owner] += SPECIES_POINTS[card.species_id]
