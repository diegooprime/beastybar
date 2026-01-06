"""Custom exception classes for the Beasty Bar simulator."""

from __future__ import annotations


class BeastyBarError(Exception):
    """Base exception for all Beasty Bar simulator errors."""


class InvalidPlayerError(BeastyBarError):
    """Raised when an invalid player index is provided."""

    def __init__(self, player: int, max_players: int) -> None:
        self.player = player
        self.max_players = max_players
        super().__init__(f"Invalid player index {player}. Must be in range [0, {max_players}).")


class InvalidActionError(BeastyBarError):
    """Raised when an invalid action is attempted."""


class InvalidHandIndexError(InvalidActionError):
    """Raised when an invalid hand index is provided."""

    def __init__(self, hand_index: int, hand_size: int) -> None:
        self.hand_index = hand_index
        self.hand_size = hand_size
        super().__init__(f"Invalid hand index {hand_index}. Hand has {hand_size} cards.")


class InvalidParametersError(InvalidActionError):
    """Raised when invalid action parameters are provided."""


class GameAlreadyTerminalError(BeastyBarError):
    """Raised when trying to step a game that has already ended."""

    def __init__(self) -> None:
        super().__init__("Cannot step; game already finished")


class InvalidSpeciesError(BeastyBarError):
    """Raised when an unknown species is referenced."""

    def __init__(self, species: str) -> None:
        self.species = species
        super().__init__(f"Unknown species: {species}")


__all__ = [
    "BeastyBarError",
    "GameAlreadyTerminalError",
    "InvalidActionError",
    "InvalidHandIndexError",
    "InvalidParametersError",
    "InvalidPlayerError",
    "InvalidSpeciesError",
]
