"""Neural network architecture for Beasty Bar policy-value network.

This module implements a Transformer-based architecture that processes
game observations and outputs policy logits (for action selection) and
a value estimate (for position evaluation).

Architecture Overview:
- Species embedding layer maps 12 species to 32-dim learned embeddings
- Positional encoding for sequence order within zones
- Transformer encoder for queue (order-sensitive)
- Set Transformer for beasty_bar/thats_it (order-invariant)
- Fusion layer with residual connections
- Policy head: 124 logits for action selection
- Value head: scalar in [-1, 1] for position evaluation

Target: 500K - 2M parameters for fast inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from _01_simulator.action_space import ACTION_DIM
from _02_agents.neural.compile import maybe_compile_network
from _01_simulator.observations import OBSERVATION_DIM

# Import NetworkConfig from utils to maintain consistency
from _02_agents.neural.utils import NetworkConfig

# Zone sizes (matching observations.py)
_NUM_SPECIES = 12  # Excluding 'unknown'
_CARD_FEATURE_DIM = 17  # Features per card
_TOTAL_CARD_SLOTS = 24  # DECK_SIZE * PLAYER_COUNT

# Zone slot counts
_QUEUE_SLOTS = 5  # MAX_QUEUE_LENGTH
_BAR_SLOTS = _TOTAL_CARD_SLOTS  # 24
_THATS_IT_SLOTS = _TOTAL_CARD_SLOTS  # 24
_OWN_HAND_SLOTS = 4  # HAND_SIZE
_OPPONENT_HAND_SLOTS = 4  # HAND_SIZE
_MASKED_CARD_DIM = 3  # Opponent hand features
_SCALARS_DIM = 7


def _get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    pe: torch.Tensor  # Registered buffer

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        result: torch.Tensor = self.dropout(x)
        return result


class CardEncoder(nn.Module):
    """Encode card features into embedding space.

    Takes raw card features and projects them to embedding dimension,
    with learned species embedding replacing one-hot encoding.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()

        # Species embedding: 12 species -> embedding_dim
        self.species_embedding = nn.Embedding(
            num_embeddings=config.num_species + 1,  # +1 for empty/padding
            embedding_dim=config.species_embedding_dim,
            padding_idx=config.num_species,  # Index for empty slots
        )

        # Project non-species features (5 dims: presence, owner, strength, points, position)
        # to match embedding dimension
        self._non_species_dim = 5
        self.feature_proj = nn.Linear(self._non_species_dim, config.species_embedding_dim)

        # Combine species embedding and projected features
        self.combine = nn.Linear(config.species_embedding_dim * 2, config.hidden_dim)

        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, card_features: torch.Tensor) -> torch.Tensor:
        """Encode card features.

        Args:
            card_features: Shape (batch, num_cards, 17) with card features:
                [presence, owner, species_onehot(12), strength, points, position]

        Returns:
            Shape (batch, num_cards, hidden_dim)
        """
        # Extract components from feature vector
        presence = card_features[:, :, 0:1]  # (batch, num_cards, 1)
        owner = card_features[:, :, 1:2]  # (batch, num_cards, 1)
        species_onehot = card_features[:, :, 2:14]  # (batch, num_cards, 12)
        strength = card_features[:, :, 14:15]  # (batch, num_cards, 1)
        points = card_features[:, :, 15:16]  # (batch, num_cards, 1)
        position = card_features[:, :, 16:17]  # (batch, num_cards, 1)

        # Convert species one-hot to index (argmax), use padding_idx for empty slots
        species_idx = species_onehot.argmax(dim=-1)  # (batch, num_cards)
        # Mark empty slots (presence=0) with padding index
        num_species = species_onehot.size(-1)
        species_idx = torch.where(
            presence.squeeze(-1) > 0.5,
            species_idx,
            torch.full_like(species_idx, num_species),
        )

        # Get species embeddings
        species_emb = self.species_embedding(species_idx)  # (batch, num_cards, embedding_dim)

        # Concatenate non-species features and project
        non_species = torch.cat([presence, owner, strength, points, position], dim=-1)
        non_species_proj = self.feature_proj(non_species)  # (batch, num_cards, embedding_dim)

        # Combine embeddings
        combined = torch.cat([species_emb, non_species_proj], dim=-1)  # (batch, num_cards, 2*embedding_dim)
        output: torch.Tensor = self.combine(combined)  # (batch, num_cards, hidden_dim)
        output = self.activation(output)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class MaskedCardEncoder(nn.Module):
    """Encode masked opponent hand cards (presence only, no species)."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(_MASKED_CARD_DIM, config.hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, masked_features: torch.Tensor) -> torch.Tensor:
        """Encode masked card features.

        Args:
            masked_features: Shape (batch, num_cards, 3)

        Returns:
            Shape (batch, num_cards, hidden_dim)
        """
        output: torch.Tensor = self.proj(masked_features)
        output = self.activation(output)
        output = self.layer_norm(output)
        return output


class TransformerZoneEncoder(nn.Module):
    """Transformer encoder for order-sensitive zones (queue).

    Uses standard transformer with positional encoding to capture
    sequential dependencies between cards.
    """

    def __init__(self, config: NetworkConfig, max_len: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.positional_encoding = PositionalEncoding(
            d_model=config.hidden_dim,
            max_len=max_len,
            dropout=config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Learnable empty zone representation (for when zone is empty)
        self.empty_rep = nn.Parameter(torch.zeros(1, config.hidden_dim))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode zone with transformer.

        Args:
            x: Shape (batch, seq_len, hidden_dim)
            padding_mask: Optional boolean mask, True = padding position

        Returns:
            Shape (batch, hidden_dim)
        """
        batch_size = x.size(0)

        # Check for completely empty zones (all padding)
        if padding_mask is not None:
            all_padded = padding_mask.all(dim=-1)  # (batch,)
        else:
            all_padded = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # For batches where all positions are padded, use empty representation
        if all_padded.all():
            return self.empty_rep.expand(batch_size, -1)

        # Apply transformer only to non-empty zones
        x = self.positional_encoding(x)

        # Handle edge case: if some but not all positions are padded
        # We need to avoid NaN from attention over all-padded sequences
        if padding_mask is not None and all_padded.any():
            # Process non-empty batches through transformer
            non_empty_mask = ~all_padded
            x_non_empty = x[non_empty_mask]
            padding_non_empty = padding_mask[non_empty_mask]

            x_encoded = self.transformer(x_non_empty, src_key_padding_mask=padding_non_empty)

            # Masked mean pooling for non-empty
            inv_mask = ~padding_non_empty  # True = valid positions
            inv_mask_expanded = inv_mask.unsqueeze(-1).float()
            pooled_non_empty = (x_encoded * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)

            # Combine with empty representation for empty batches
            result = self.empty_rep.expand(batch_size, -1).clone()
            result[non_empty_mask] = pooled_non_empty
            return result

        # Normal case: no completely empty zones
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Masked mean pooling
        if padding_mask is not None:
            inv_mask = ~padding_mask  # True = valid positions
            inv_mask_expanded = inv_mask.unsqueeze(-1).float()
            pooled = (x * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return pooled


class SetTransformerEncoder(nn.Module):
    """Set Transformer for order-invariant zones (beasty_bar, thats_it).

    Uses self-attention without positional encoding to process cards
    in a permutation-invariant manner.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Learnable empty zone representation (for when zone is empty)
        self.empty_rep = nn.Parameter(torch.zeros(1, config.hidden_dim))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode zone with set transformer.

        Args:
            x: Shape (batch, seq_len, hidden_dim)
            padding_mask: Optional boolean mask, True = padding position

        Returns:
            Shape (batch, hidden_dim)
        """
        batch_size = x.size(0)

        # Check for completely empty zones (all padding)
        if padding_mask is not None:
            all_padded = padding_mask.all(dim=-1)  # (batch,)
        else:
            all_padded = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        # For batches where all positions are padded, use empty representation
        if all_padded.all():
            return self.empty_rep.expand(batch_size, -1)

        # Handle edge case: if some but not all batches are completely padded
        if padding_mask is not None and all_padded.any():
            # Process non-empty batches through transformer
            non_empty_mask = ~all_padded
            x_non_empty = x[non_empty_mask]
            padding_non_empty = padding_mask[non_empty_mask]

            x_encoded = self.transformer(x_non_empty, src_key_padding_mask=padding_non_empty)

            # Masked mean pooling for non-empty
            inv_mask = ~padding_non_empty  # True = valid positions
            inv_mask_expanded = inv_mask.unsqueeze(-1).float()
            pooled_non_empty = (x_encoded * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)

            # Combine with empty representation for empty batches
            result = self.empty_rep.expand(batch_size, -1).clone()
            result[non_empty_mask] = pooled_non_empty
            return result

        # Normal case: no completely empty zones
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Masked mean pooling (simpler and more stable than attention pooling)
        if padding_mask is not None:
            inv_mask = ~padding_mask  # True = valid positions
            inv_mask_expanded = inv_mask.unsqueeze(-1).float()
            pooled = (x * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return pooled


class FusionBlock(nn.Module):
    """Residual fusion block with FC layers."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        self.residual_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fusion block with residual connection.

        Args:
            x: Shape (batch, input_dim)

        Returns:
            Shape (batch, hidden_dim)
        """
        residual: torch.Tensor = self.residual_proj(x)

        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x + residual


class BeastyBarNetwork(nn.Module):
    """Policy-value network for Beasty Bar.

    Architecture:
    - Card encoders convert raw features to embeddings
    - Zone encoders (Transformer/SetTransformer) process each zone
    - Fusion layer combines zone representations
    - Policy head outputs 124 action logits
    - Value head outputs scalar position evaluation

    Input: Flattened 988-dim observation tensor
    Output: (policy_logits, value)
        - policy_logits: (batch, 124) raw logits (mask applied externally)
        - value: (batch, 1) scalar in [-1, 1]
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()

        # Validate dimensions match expectations
        if self.config.observation_dim != OBSERVATION_DIM:
            raise ValueError(
                f"Config observation_dim ({self.config.observation_dim}) "
                f"does not match OBSERVATION_DIM ({OBSERVATION_DIM})"
            )
        if self.config.action_dim != ACTION_DIM:
            raise ValueError(
                f"Config action_dim ({self.config.action_dim}) "
                f"does not match ACTION_DIM ({ACTION_DIM})"
            )

        # Card encoders
        self.card_encoder = CardEncoder(self.config)
        self.masked_card_encoder = MaskedCardEncoder(self.config)

        # Zone encoders
        self.queue_encoder = TransformerZoneEncoder(
            self.config, max_len=self.config.max_queue_length
        )
        self.bar_encoder = SetTransformerEncoder(self.config)
        self.thats_it_encoder = SetTransformerEncoder(self.config)
        self.hand_encoder = SetTransformerEncoder(self.config)
        self.opponent_hand_encoder = SetTransformerEncoder(self.config)

        # Scalar feature projection
        self.scalar_proj = nn.Linear(_SCALARS_DIM, self.config.hidden_dim)

        # Fusion: 5 zone representations + scalars = 6 * hidden_dim
        fusion_input_dim = 6 * self.config.hidden_dim

        # Fusion blocks (3 layers with residual connections)
        self.fusion1 = FusionBlock(fusion_input_dim, self.config.hidden_dim, self.config.dropout)
        self.fusion2 = FusionBlock(
            self.config.hidden_dim, self.config.hidden_dim, self.config.dropout
        )
        self.fusion3 = FusionBlock(
            self.config.hidden_dim, self.config.hidden_dim, self.config.dropout
        )

        # Policy head: hidden_dim -> ACTION_DIM logits
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.action_dim),
        )

        # Value head: hidden_dim -> 1 (with tanh)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Tanh(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights with appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _parse_observation(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flattened observation into zone tensors.

        Args:
            obs: Shape (batch, 988) flattened observation

        Returns:
            Tuple of (queue, bar, thats_it, hand, opponent_hand, scalars)
        """
        batch_size = obs.size(0)
        offset = 0

        card_feat_dim = self.config.card_feature_dim

        # Queue: 5 * 17 = 85
        queue_slots = self.config.max_queue_length
        queue_flat = obs[:, offset : offset + queue_slots * card_feat_dim]
        queue = queue_flat.view(batch_size, queue_slots, card_feat_dim)
        offset += queue_slots * card_feat_dim

        # Beasty Bar: 24 * 17 = 408
        bar_slots = self.config.max_bar_length
        bar_flat = obs[:, offset : offset + bar_slots * card_feat_dim]
        bar = bar_flat.view(batch_size, bar_slots, card_feat_dim)
        offset += bar_slots * card_feat_dim

        # That's It: 24 * 17 = 408
        thats_it_flat = obs[:, offset : offset + bar_slots * card_feat_dim]
        thats_it = thats_it_flat.view(batch_size, bar_slots, card_feat_dim)
        offset += bar_slots * card_feat_dim

        # Own Hand: 4 * 17 = 68
        hand_slots = self.config.hand_size
        hand_flat = obs[:, offset : offset + hand_slots * card_feat_dim]
        hand = hand_flat.view(batch_size, hand_slots, card_feat_dim)
        offset += hand_slots * card_feat_dim

        # Opponent Hand (masked): 4 * 3 = 12
        opponent_flat = obs[:, offset : offset + hand_slots * _MASKED_CARD_DIM]
        opponent_hand = opponent_flat.view(batch_size, hand_slots, _MASKED_CARD_DIM)
        offset += hand_slots * _MASKED_CARD_DIM

        # Scalars: 7
        scalars = obs[:, offset : offset + _SCALARS_DIM]

        return queue, bar, thats_it, hand, opponent_hand, scalars

    def _create_padding_mask(self, card_features: torch.Tensor) -> torch.Tensor:
        """Create padding mask from card presence feature.

        Args:
            card_features: Shape (batch, num_cards, feat_dim) with presence at index 0

        Returns:
            Boolean tensor (batch, num_cards), True = padding position
        """
        presence = card_features[:, :, 0]  # (batch, num_cards)
        return presence < 0.5  # True where cards are not present

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            obs: Flattened observation tensor, shape (batch, 988) or (988,)
            mask: Optional action mask, shape (batch, 124) or (124,)
                  Not used for masking here; returned logits are raw.
                  Masking should be applied externally for flexibility.

        Returns:
            Tuple of:
                - policy_logits: Shape (batch, 124) raw action logits
                - value: Shape (batch, 1) position evaluation in [-1, 1]
        """
        # Handle unbatched input
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Parse observation into zones
        queue, bar, thats_it, hand, opponent_hand, scalars = self._parse_observation(obs)

        # Create padding masks
        queue_mask = self._create_padding_mask(queue)
        bar_mask = self._create_padding_mask(bar)
        thats_it_mask = self._create_padding_mask(thats_it)
        hand_mask = self._create_padding_mask(hand)
        opponent_mask = opponent_hand[:, :, 0] < 0.5  # Presence is index 0 for masked cards

        # Encode cards
        queue_encoded = self.card_encoder(queue)
        bar_encoded = self.card_encoder(bar)
        thats_it_encoded = self.card_encoder(thats_it)
        hand_encoded = self.card_encoder(hand)
        opponent_encoded = self.masked_card_encoder(opponent_hand)

        # Encode zones
        queue_rep = self.queue_encoder(queue_encoded, padding_mask=queue_mask)
        bar_rep = self.bar_encoder(bar_encoded, padding_mask=bar_mask)
        thats_it_rep = self.thats_it_encoder(thats_it_encoded, padding_mask=thats_it_mask)
        hand_rep = self.hand_encoder(hand_encoded, padding_mask=hand_mask)
        opponent_rep = self.opponent_hand_encoder(opponent_encoded, padding_mask=opponent_mask)

        # Project scalars
        scalar_rep = self.scalar_proj(scalars)
        scalar_rep = F.gelu(scalar_rep)

        # Concatenate all representations
        fused = torch.cat(
            [queue_rep, bar_rep, thats_it_rep, hand_rep, opponent_rep, scalar_rep],
            dim=-1,
        )

        # Apply fusion blocks
        fused = self.fusion1(fused)
        fused = self.fusion2(fused)
        fused = self.fusion3(fused)

        # Policy and value heads
        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)

        # Handle squeeze for unbatched input
        if squeeze_output:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)

        return policy_logits, value

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_network(
    config: NetworkConfig | None = None,
    *,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    species_embedding_dim: int | None = None,
) -> BeastyBarNetwork:
    """Factory function to create BeastyBarNetwork with configurable parameters.

    Args:
        config: NetworkConfig to use. If provided, other args override specific fields.
        hidden_dim: Override hidden dimension
        num_heads: Override number of attention heads
        num_layers: Override number of encoder layers
        dropout: Override dropout probability
        species_embedding_dim: Override species embedding dimension

    Returns:
        Configured BeastyBarNetwork instance
    """
    if config is None:
        config = NetworkConfig()

    # Override specific fields if provided
    overrides: dict[str, int | float] = {}
    if hidden_dim is not None:
        overrides["hidden_dim"] = hidden_dim
    if num_heads is not None:
        overrides["num_heads"] = num_heads
    if num_layers is not None:
        overrides["num_layers"] = num_layers
    if dropout is not None:
        overrides["dropout"] = dropout
    if species_embedding_dim is not None:
        overrides["species_embedding_dim"] = species_embedding_dim

    if overrides:
        # Create new config with overrides
        # Use explicit casts to ensure correct types
        config = NetworkConfig(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dim=int(overrides.get("hidden_dim", config.hidden_dim)),
            num_heads=int(overrides.get("num_heads", config.num_heads)),
            num_layers=int(overrides.get("num_layers", config.num_layers)),
            dropout=float(overrides.get("dropout", config.dropout)),
            species_embedding_dim=int(overrides.get("species_embedding_dim", config.species_embedding_dim)),
            card_feature_dim=config.card_feature_dim,
            num_species=config.num_species,
            max_queue_length=config.max_queue_length,
            max_bar_length=config.max_bar_length,
            hand_size=config.hand_size,
        )

    return BeastyBarNetwork(config)


__all__ = [
    "BeastyBarNetwork",
    "create_network",
    "maybe_compile_network",
]
