"""BeastyBarNetworkV2: Enhanced architecture for superhuman play.

This module implements the Phase 3 architecture improvements from ROADMAP_TO_SUPERHUMAN.md:

1. **Asymmetric Zone Encoders**
   - Queue encoder: 6 layers (order-sensitive, critical for predicting wins)
   - Bar/ThatsIt encoders: 2 layers (order-invariant, less critical)
   - Hand encoders: 2 layers

2. **Dueling Architecture**
   - Separate value stream (state value V(s))
   - Separate advantage stream (action advantages A(s,a))
   - Combined: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
   - Improves value estimation stability

3. **Auxiliary Prediction Heads**
   - Queue position predictor: Predict final queue positions of cards
   - Score predictor: Predict final score margin
   - Multi-task learning improves representation quality

Target: ~3.5M parameters (up from ~1.3M in V1)

Example:
    config = NetworkConfigV2(
        hidden_dim=256,
        queue_layers=6,
        bar_layers=2,
    )
    network = BeastyBarNetworkV2(config)
    policy_logits, value, aux_outputs = network(obs, return_aux=True)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM
from _02_agents.neural.compile import maybe_compile_network

# Zone sizes (matching observations.py)
_NUM_SPECIES = 12
_CARD_FEATURE_DIM = 17
_TOTAL_CARD_SLOTS = 24

# Zone slot counts
_QUEUE_SLOTS = 5
_BAR_SLOTS = _TOTAL_CARD_SLOTS
_THATS_IT_SLOTS = _TOTAL_CARD_SLOTS
_OWN_HAND_SLOTS = 4
_OPPONENT_HAND_SLOTS = 4
_MASKED_CARD_DIM = 3
_SCALARS_DIM = 7


# ============================================================================
# Network Configuration V2
# ============================================================================


@dataclass(frozen=True)
class NetworkConfigV2:
    """Configuration for BeastyBarNetworkV2 with asymmetric encoders.

    Key differences from NetworkConfig (V1):
    - Separate layer counts for each zone type
    - Fusion layer depth is configurable
    - Auxiliary head configuration
    - Dueling architecture settings

    Attributes:
        observation_dim: Input dimension (988 for Beasty Bar).
        action_dim: Output dimension for policy head (124).
        hidden_dim: Size of hidden layers in network.
        num_heads: Number of attention heads in transformer layers.
        queue_layers: Number of transformer layers for queue encoder.
        bar_layers: Number of transformer layers for bar/thats_it encoders.
        hand_layers: Number of transformer layers for hand encoders.
        fusion_layers: Number of residual blocks in fusion network.
        dropout: Dropout probability for regularization.
        species_embedding_dim: Dimension of learned species embeddings.
        card_feature_dim: Dimension of per-card feature vector.
        num_species: Number of distinct species.
        max_queue_length: Maximum cards in queue zone.
        max_bar_length: Maximum cards in bar/thats_it zones.
        hand_size: Maximum cards in hand.
        use_dueling: Enable dueling architecture for policy head.
        use_auxiliary_heads: Enable auxiliary prediction heads.
        auxiliary_weight: Weight for auxiliary losses in training.
    """

    # Core dimensions
    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM

    # Network architecture - asymmetric depths
    hidden_dim: int = 256
    num_heads: int = 8
    queue_layers: int = 6  # More capacity for order-sensitive queue
    bar_layers: int = 2    # Less for order-invariant zones
    hand_layers: int = 2
    fusion_layers: int = 4
    dropout: float = 0.1
    species_embedding_dim: int = 64

    # Card encoding
    card_feature_dim: int = 17
    num_species: int = 12

    # Zone sizes
    max_queue_length: int = 5
    max_bar_length: int = 24
    hand_size: int = 4

    # Dueling architecture
    use_dueling: bool = True

    # Auxiliary heads for multi-task learning
    use_auxiliary_heads: bool = True
    auxiliary_weight: float = 0.1  # Weight for aux losses

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkConfigV2:
        """Create config from dictionary."""
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


# ============================================================================
# Building Blocks
# ============================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """Residual block with pre-LayerNorm."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class CardEncoderV2(nn.Module):
    """Enhanced card encoder with species embeddings."""

    def __init__(self, config: NetworkConfigV2) -> None:
        super().__init__()

        self.species_embedding = nn.Embedding(
            num_embeddings=config.num_species + 1,
            embedding_dim=config.species_embedding_dim,
            padding_idx=config.num_species,
        )

        self._non_species_dim = 5
        self.feature_proj = nn.Linear(self._non_species_dim, config.species_embedding_dim)
        self.combine = nn.Linear(config.species_embedding_dim * 2, config.hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, card_features: torch.Tensor) -> torch.Tensor:
        presence = card_features[:, :, 0:1]
        owner = card_features[:, :, 1:2]
        species_onehot = card_features[:, :, 2:14]
        strength = card_features[:, :, 14:15]
        points = card_features[:, :, 15:16]
        position = card_features[:, :, 16:17]

        species_idx = species_onehot.argmax(dim=-1)
        num_species = species_onehot.size(-1)
        species_idx = torch.where(
            presence.squeeze(-1) > 0.5,
            species_idx,
            torch.full_like(species_idx, num_species),
        )

        species_emb = self.species_embedding(species_idx)
        non_species = torch.cat([presence, owner, strength, points, position], dim=-1)
        non_species_proj = self.feature_proj(non_species)

        combined = torch.cat([species_emb, non_species_proj], dim=-1)
        output = self.combine(combined)
        output = self.activation(output)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class MaskedCardEncoderV2(nn.Module):
    """Encode masked opponent hand cards."""

    def __init__(self, config: NetworkConfigV2) -> None:
        super().__init__()
        self.proj = nn.Linear(_MASKED_CARD_DIM, config.hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, masked_features: torch.Tensor) -> torch.Tensor:
        output = self.proj(masked_features)
        output = self.activation(output)
        output = self.layer_norm(output)
        return output


class AsymmetricTransformerEncoder(nn.Module):
    """Transformer encoder with configurable depth for asymmetric architecture.

    This encoder supports both order-sensitive (with positional encoding) and
    order-invariant (without positional encoding) processing modes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        max_len: int = 64,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding

        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=hidden_dim, max_len=max_len, dropout=dropout
            )
        else:
            self.positional_encoding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.empty_rep = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with CUDA graph compatible operations (no dynamic indexing).

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            padding_mask: Optional mask of shape (batch, seq_len), True = padded.

        Returns:
            Pooled representation of shape (batch, hidden_dim).
        """
        batch_size = x.size(0)

        # Apply positional encoding if enabled
        if self.use_positional_encoding and self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Always process full batch - transformer handles masking internally
        # This avoids dynamic boolean indexing which breaks CUDA graphs
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Vectorized pooling (no conditional indexing - CUDA graph safe)
        if padding_mask is not None:
            inv_mask = ~padding_mask
            inv_mask_expanded = inv_mask.unsqueeze(-1).float()
            # Masked positions contribute 0, then we normalize
            pooled = (x * inv_mask_expanded).sum(dim=1) / inv_mask_expanded.sum(dim=1).clamp(min=1)
            # Handle all-padded case: use empty_rep
            all_padded = padding_mask.all(dim=-1, keepdim=True)
            pooled = torch.where(all_padded, self.empty_rep.expand(batch_size, -1), pooled)
        else:
            pooled = x.mean(dim=1)

        return pooled


# ============================================================================
# Dueling Architecture
# ============================================================================


class DuelingHead(nn.Module):
    """Dueling architecture combining state value and action advantages.

    Output: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

    This separation helps:
    - Learn state values more accurately
    - Identify advantage of specific actions
    - More stable value estimation
    """

    def __init__(self, hidden_dim: int, action_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Value stream: predicts V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: predicts A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Q-values using dueling decomposition.

        Args:
            x: Hidden state of shape (batch, hidden_dim).
            action_mask: Optional mask of shape (batch, action_dim).

        Returns:
            Policy logits of shape (batch, action_dim).
        """
        value = self.value_stream(x)  # (batch, 1)
        advantage = self.advantage_stream(x)  # (batch, action_dim)

        # Mask advantages for illegal actions before computing mean
        # Use torch.where instead of boolean indexing (CUDA graph safe)
        if action_mask is not None:
            # Compute mean only over legal actions
            legal_count = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            advantage_masked = torch.where(action_mask > 0, advantage, torch.zeros_like(advantage))
            advantage_mean = advantage_masked.sum(dim=-1, keepdim=True) / legal_count
        else:
            advantage_mean = advantage.mean(dim=-1, keepdim=True)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage_mean

        return q_values


# ============================================================================
# Auxiliary Prediction Heads
# ============================================================================


class AuxiliaryHeads(nn.Module):
    """Auxiliary prediction heads for multi-task learning.

    Predicting auxiliary targets improves representation quality:
    - Queue position predictor: Where will each card end up?
    - Score margin predictor: Final point difference

    These auxiliary tasks provide additional gradient signal and
    encourage the network to learn more useful representations.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Predict final queue positions (which cards survive)
        # 5 positions in queue, predict probability distribution
        self.queue_position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, _QUEUE_SLOTS),
        )

        # Predict final score margin (points difference)
        # Output is scalar, trained with MSE
        self.score_margin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Predict number of cards that will enter beasty bar
        # Useful for strategic planning
        self.cards_to_bar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute auxiliary predictions.

        Args:
            x: Hidden state of shape (batch, hidden_dim).

        Returns:
            Dictionary with auxiliary predictions:
            - queue_position_logits: (batch, 5) logits for queue positions
            - score_margin: (batch, 1) predicted score margin
            - cards_to_bar: (batch, 1) predicted cards entering bar
        """
        return {
            "queue_position_logits": self.queue_position_head(x),
            "score_margin": self.score_margin_head(x),
            "cards_to_bar": self.cards_to_bar_head(x),
        }


# ============================================================================
# Main Network V2
# ============================================================================


class BeastyBarNetworkV2(nn.Module):
    """Enhanced policy-value network with asymmetric encoders and dueling architecture.

    Architecture improvements over V1:
    1. Asymmetric zone encoders - more capacity where it matters (queue)
    2. Dueling architecture - separate value and advantage streams
    3. Auxiliary heads - multi-task learning for better representations
    4. Deeper fusion network with more residual blocks

    Input: Flattened 988-dim observation tensor
    Output: (policy_logits, value) and optionally auxiliary predictions

    Parameters: ~3.5M (target) vs ~1.3M in V1
    """

    def __init__(self, config: NetworkConfigV2 | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfigV2()

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
        self.card_encoder = CardEncoderV2(self.config)
        self.masked_card_encoder = MaskedCardEncoderV2(self.config)

        # Asymmetric zone encoders
        # Queue: order-sensitive, most critical - 6 layers
        self.queue_encoder = AsymmetricTransformerEncoder(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.queue_layers,
            dropout=self.config.dropout,
            max_len=self.config.max_queue_length,
            use_positional_encoding=True,  # Order matters
        )

        # Bar/ThatsIt: order-invariant, less critical - 2 layers
        self.bar_encoder = AsymmetricTransformerEncoder(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.bar_layers,
            dropout=self.config.dropout,
            max_len=self.config.max_bar_length,
            use_positional_encoding=False,  # Order doesn't matter
        )
        self.thats_it_encoder = AsymmetricTransformerEncoder(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.bar_layers,
            dropout=self.config.dropout,
            max_len=self.config.max_bar_length,
            use_positional_encoding=False,
        )

        # Hand encoders - 2 layers
        self.hand_encoder = AsymmetricTransformerEncoder(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.hand_layers,
            dropout=self.config.dropout,
            max_len=self.config.hand_size,
            use_positional_encoding=False,  # Hand order doesn't matter
        )
        self.opponent_hand_encoder = AsymmetricTransformerEncoder(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.hand_layers,
            dropout=self.config.dropout,
            max_len=self.config.hand_size,
            use_positional_encoding=False,
        )

        # Scalar feature projection
        self.scalar_proj = nn.Sequential(
            nn.Linear(_SCALARS_DIM, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
        )

        # Fusion: 5 zone representations + scalars = 6 * hidden_dim
        fusion_input_dim = 6 * self.config.hidden_dim

        # Initial projection to hidden_dim
        self.fusion_proj = nn.Linear(fusion_input_dim, self.config.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.config.hidden_dim)

        # Fusion residual blocks
        self.fusion_blocks = nn.ModuleList([
            ResidualBlock(self.config.hidden_dim, self.config.dropout)
            for _ in range(self.config.fusion_layers)
        ])

        # Policy head: either dueling or standard
        if self.config.use_dueling:
            self.policy_head = DuelingHead(
                hidden_dim=self.config.hidden_dim,
                action_dim=self.config.action_dim,
                dropout=self.config.dropout,
            )
        else:
            self.policy_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim // 2, self.config.action_dim),
            )

        # Value head: deep with residual blocks
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            ResidualBlock(self.config.hidden_dim, self.config.dropout),
            ResidualBlock(self.config.hidden_dim, self.config.dropout),
            ResidualBlock(self.config.hidden_dim, self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Tanh(),
        )

        # Auxiliary heads for multi-task learning
        if self.config.use_auxiliary_heads:
            self.auxiliary_heads = AuxiliaryHeads(
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )
        else:
            self.auxiliary_heads = None

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
        """Parse flattened observation into zone tensors."""
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
        """Create padding mask from card presence feature."""
        presence = card_features[:, :, 0]
        return presence < 0.5

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            obs: Flattened observation tensor, shape (batch, 988) or (988,).
            mask: Optional action mask, shape (batch, 124) or (124,).
            return_aux: If True, return auxiliary predictions.

        Returns:
            If return_aux is False:
                Tuple of (policy_logits, value)
            If return_aux is True:
                Tuple of (policy_logits, value, aux_outputs)
        """
        # Handle unbatched input
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        # Parse observation into zones
        queue, bar, thats_it, hand, opponent_hand, scalars = self._parse_observation(obs)

        # Create padding masks
        queue_mask = self._create_padding_mask(queue)
        bar_mask = self._create_padding_mask(bar)
        thats_it_mask = self._create_padding_mask(thats_it)
        hand_mask = self._create_padding_mask(hand)
        opponent_mask = opponent_hand[:, :, 0] < 0.5

        # Encode cards
        queue_encoded = self.card_encoder(queue)
        bar_encoded = self.card_encoder(bar)
        thats_it_encoded = self.card_encoder(thats_it)
        hand_encoded = self.card_encoder(hand)
        opponent_encoded = self.masked_card_encoder(opponent_hand)

        # Encode zones with asymmetric encoders
        queue_rep = self.queue_encoder(queue_encoded, padding_mask=queue_mask)
        bar_rep = self.bar_encoder(bar_encoded, padding_mask=bar_mask)
        thats_it_rep = self.thats_it_encoder(thats_it_encoded, padding_mask=thats_it_mask)
        hand_rep = self.hand_encoder(hand_encoded, padding_mask=hand_mask)
        opponent_rep = self.opponent_hand_encoder(opponent_encoded, padding_mask=opponent_mask)

        # Project scalars
        scalar_rep = self.scalar_proj(scalars)

        # Concatenate all representations
        fused = torch.cat(
            [queue_rep, bar_rep, thats_it_rep, hand_rep, opponent_rep, scalar_rep],
            dim=-1,
        )

        # Fusion projection
        fused = self.fusion_proj(fused)
        fused = F.gelu(fused)
        fused = self.fusion_norm(fused)

        # Apply fusion blocks
        for block in self.fusion_blocks:
            fused = block(fused)

        # Policy head
        if self.config.use_dueling:
            policy_logits = self.policy_head(fused, action_mask=mask)
        else:
            policy_logits = self.policy_head(fused)

        # Value head
        value = self.value_head(fused)

        # Handle squeeze for unbatched input
        if squeeze_output:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)

        if return_aux and self.auxiliary_heads is not None:
            if squeeze_output:
                aux_outputs = self.auxiliary_heads(fused.unsqueeze(0))
                aux_outputs = {k: v.squeeze(0) for k, v in aux_outputs.items()}
            else:
                aux_outputs = self.auxiliary_heads(fused)
            return policy_logits, value, aux_outputs

        return policy_logits, value

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder_parameter_counts(self) -> dict[str, int]:
        """Get parameter counts for each encoder component."""
        counts = {}

        # Count card encoder
        counts["card_encoder"] = sum(
            p.numel() for p in self.card_encoder.parameters() if p.requires_grad
        )
        counts["masked_card_encoder"] = sum(
            p.numel() for p in self.masked_card_encoder.parameters() if p.requires_grad
        )

        # Count zone encoders
        counts["queue_encoder"] = sum(
            p.numel() for p in self.queue_encoder.parameters() if p.requires_grad
        )
        counts["bar_encoder"] = sum(
            p.numel() for p in self.bar_encoder.parameters() if p.requires_grad
        )
        counts["thats_it_encoder"] = sum(
            p.numel() for p in self.thats_it_encoder.parameters() if p.requires_grad
        )
        counts["hand_encoder"] = sum(
            p.numel() for p in self.hand_encoder.parameters() if p.requires_grad
        )
        counts["opponent_hand_encoder"] = sum(
            p.numel() for p in self.opponent_hand_encoder.parameters() if p.requires_grad
        )

        # Count fusion
        fusion_params = sum(p.numel() for p in self.fusion_proj.parameters() if p.requires_grad)
        fusion_params += sum(p.numel() for p in self.fusion_norm.parameters() if p.requires_grad)
        for block in self.fusion_blocks:
            fusion_params += sum(p.numel() for p in block.parameters() if p.requires_grad)
        counts["fusion"] = fusion_params

        # Count heads
        counts["policy_head"] = sum(
            p.numel() for p in self.policy_head.parameters() if p.requires_grad
        )
        counts["value_head"] = sum(
            p.numel() for p in self.value_head.parameters() if p.requires_grad
        )

        if self.auxiliary_heads is not None:
            counts["auxiliary_heads"] = sum(
                p.numel() for p in self.auxiliary_heads.parameters() if p.requires_grad
            )

        return counts


def create_network_v2(
    config: NetworkConfigV2 | None = None,
    **kwargs: Any,
) -> BeastyBarNetworkV2:
    """Factory function to create BeastyBarNetworkV2.

    Args:
        config: NetworkConfigV2 to use.
        **kwargs: Override config fields.

    Returns:
        Configured BeastyBarNetworkV2 instance.
    """
    if config is None:
        config = NetworkConfigV2()

    if kwargs:
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        config = NetworkConfigV2.from_dict(config_dict)

    return BeastyBarNetworkV2(config)


__all__ = [
    "AsymmetricTransformerEncoder",
    "AuxiliaryHeads",
    "BeastyBarNetworkV2",
    "DuelingHead",
    "NetworkConfigV2",
    "create_network_v2",
    "maybe_compile_network",
]
