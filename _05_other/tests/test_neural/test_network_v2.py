"""Test suite for BeastyBarNetworkV2 architecture.

Tests forward pass, output shapes, parameter counts, dueling architecture,
auxiliary heads, and asymmetric encoder configurations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from _01_simulator.action_space import ACTION_DIM
from _01_simulator.observations import OBSERVATION_DIM, state_to_tensor
from _01_simulator import state
from _02_agents.neural.network_v2 import (
    BeastyBarNetworkV2,
    NetworkConfigV2,
    create_network_v2,
    DuelingHead,
    AuxiliaryHeads,
    AsymmetricTransformerEncoder,
)


@pytest.fixture
def network():
    """Create a small test network V2."""
    config = NetworkConfigV2(
        hidden_dim=64,
        num_heads=4,
        queue_layers=2,
        bar_layers=1,
        hand_layers=1,
        fusion_layers=2,
        dropout=0.0,
        use_dueling=True,
        use_auxiliary_heads=True,
    )
    return create_network_v2(config)


@pytest.fixture
def network_no_aux():
    """Create a test network V2 without auxiliary heads."""
    config = NetworkConfigV2(
        hidden_dim=64,
        num_heads=4,
        queue_layers=2,
        bar_layers=1,
        hand_layers=1,
        fusion_layers=2,
        dropout=0.0,
        use_dueling=True,
        use_auxiliary_heads=False,
    )
    return create_network_v2(config)


class TestNetworkV2ForwardPass:
    """Test forward pass through NetworkV2."""

    def test_forward_shapes(self, network):
        """Verify network forward pass produces correct output shapes."""
        batch_size = 4
        obs = torch.randn(batch_size, OBSERVATION_DIM)

        policy_logits, value = network(obs)

        assert policy_logits.shape == (batch_size, ACTION_DIM), \
            f"Expected policy shape (4, {ACTION_DIM}), got {policy_logits.shape}"
        assert value.shape == (batch_size, 1), \
            f"Expected value shape (4, 1), got {value.shape}"

    def test_forward_unbatched(self, network):
        """Test single input (unbatched) forward pass."""
        obs_single = torch.randn(OBSERVATION_DIM)
        policy_single, value_single = network(obs_single)

        assert policy_single.shape == (ACTION_DIM,), \
            f"Expected single policy shape ({ACTION_DIM},), got {policy_single.shape}"
        assert value_single.dim() == 0 or value_single.shape == (1,), \
            f"Expected single value to be scalar, got {value_single.shape}"

    def test_forward_with_aux(self, network):
        """Test forward pass with auxiliary outputs."""
        batch_size = 4
        obs = torch.randn(batch_size, OBSERVATION_DIM)

        policy_logits, value, aux_outputs = network(obs, return_aux=True)

        assert "queue_position_logits" in aux_outputs
        assert "score_margin" in aux_outputs
        assert "cards_to_bar" in aux_outputs

        assert aux_outputs["queue_position_logits"].shape == (batch_size, 5)
        assert aux_outputs["score_margin"].shape == (batch_size, 1)
        assert aux_outputs["cards_to_bar"].shape == (batch_size, 1)

    def test_forward_no_aux_network(self, network_no_aux):
        """Test forward pass without auxiliary heads returns only policy and value."""
        obs = torch.randn(4, OBSERVATION_DIM)
        result = network_no_aux(obs, return_aux=True)

        # Should return tuple of 2 (no aux heads)
        assert len(result) == 2


class TestNetworkV2ValueRange:
    """Test value output constraints."""

    def test_value_in_range(self, network):
        """Verify value estimates are in [-1, 1] range due to tanh."""
        batch_size = 10
        obs = torch.randn(batch_size, OBSERVATION_DIM)

        network.eval()
        with torch.no_grad():
            _, value = network(obs)

        assert torch.all(value >= -1.0), f"Value has elements < -1.0: min={value.min()}"
        assert torch.all(value <= 1.0), f"Value has elements > 1.0: max={value.max()}"


class TestNetworkV2Determinism:
    """Test deterministic behavior in eval mode."""

    def test_deterministic_eval(self, network):
        """Same input produces same output in eval mode."""
        network.eval()

        torch.manual_seed(42)
        obs = torch.randn(2, OBSERVATION_DIM)

        with torch.no_grad():
            policy1, value1 = network(obs)
            policy2, value2 = network(obs)

        assert torch.allclose(policy1, policy2), "Policy outputs differ"
        assert torch.allclose(value1, value2), "Value outputs differ"


class TestNetworkV2ParameterCount:
    """Test parameter counting and scaling."""

    def test_parameter_count_basic(self, network):
        """Verify parameter count method works correctly."""
        param_count = network.count_parameters()

        assert param_count > 0, "Network should have parameters"
        assert param_count < 50_000_000, f"Parameter count {param_count:,} seems too large"

    def test_asymmetric_encoder_counts(self, network):
        """Verify asymmetric encoder parameter distribution."""
        counts = network.get_encoder_parameter_counts()

        # Queue encoder should have more parameters than bar encoder
        # because it has more layers (2 vs 1 in test config)
        assert counts["queue_encoder"] > counts["bar_encoder"], \
            "Queue encoder should have more params than bar encoder"

    def test_full_scale_parameter_count(self):
        """Test parameter count for production-scale network."""
        config = NetworkConfigV2(
            hidden_dim=256,
            num_heads=8,
            queue_layers=6,
            bar_layers=2,
            hand_layers=2,
            fusion_layers=4,
        )
        network = BeastyBarNetworkV2(config)
        param_count = network.count_parameters()

        # V2 with asymmetric encoders (6/2/2) should be ~10-15M params
        # This is more efficient than V1 with uniform 4 layers (~17M params)
        assert 8_000_000 < param_count < 20_000_000, \
            f"Full-scale V2 network should have 8-20M params, got {param_count:,}"

    def test_v2_more_efficient_than_v1_equivalent(self):
        """Verify V2 is more parameter-efficient than equivalent V1."""
        from _02_agents.neural.network import BeastyBarNetwork
        from _02_agents.neural.utils import NetworkConfig

        # V1 with 4 uniform layers
        v1_config = NetworkConfig(hidden_dim=256, num_heads=8, num_layers=4)
        v1_network = BeastyBarNetwork(v1_config)

        # V2 with asymmetric layers (more queue capacity, less elsewhere)
        v2_config = NetworkConfigV2(
            hidden_dim=256,
            num_heads=8,
            queue_layers=6,  # More capacity for queue
            bar_layers=2,
            hand_layers=2,
            fusion_layers=4,
        )
        v2_network = BeastyBarNetworkV2(v2_config)

        # V2 should have fewer total parameters despite more queue capacity
        assert v2_network.count_parameters() < v1_network.count_parameters(), \
            "V2 should be more parameter-efficient than V1"


class TestDuelingArchitecture:
    """Test dueling head component."""

    def test_dueling_head_output_shape(self):
        """Test dueling head produces correct output shape."""
        hidden_dim = 64
        action_dim = 124
        batch_size = 8

        head = DuelingHead(hidden_dim, action_dim)
        x = torch.randn(batch_size, hidden_dim)

        output = head(x)
        assert output.shape == (batch_size, action_dim)

    def test_dueling_head_with_mask(self):
        """Test dueling head with action mask."""
        hidden_dim = 64
        action_dim = 124
        batch_size = 8

        head = DuelingHead(hidden_dim, action_dim)
        x = torch.randn(batch_size, hidden_dim)
        mask = torch.zeros(batch_size, action_dim)
        mask[:, :10] = 1.0  # Only first 10 actions are legal

        output = head(x, action_mask=mask)
        assert output.shape == (batch_size, action_dim)

    def test_dueling_vs_non_dueling(self):
        """Compare dueling and non-dueling networks."""
        config_dueling = NetworkConfigV2(
            hidden_dim=64,
            queue_layers=2,
            bar_layers=1,
            use_dueling=True,
        )
        config_no_dueling = NetworkConfigV2(
            hidden_dim=64,
            queue_layers=2,
            bar_layers=1,
            use_dueling=False,
        )

        net_dueling = BeastyBarNetworkV2(config_dueling)
        net_no_dueling = BeastyBarNetworkV2(config_no_dueling)

        obs = torch.randn(4, OBSERVATION_DIM)

        policy_d, value_d = net_dueling(obs)
        policy_nd, value_nd = net_no_dueling(obs)

        # Both should produce valid shapes
        assert policy_d.shape == policy_nd.shape
        assert value_d.shape == value_nd.shape


class TestAuxiliaryHeads:
    """Test auxiliary prediction heads."""

    def test_auxiliary_head_shapes(self):
        """Test auxiliary head output shapes."""
        hidden_dim = 64
        batch_size = 8

        heads = AuxiliaryHeads(hidden_dim)
        x = torch.randn(batch_size, hidden_dim)

        outputs = heads(x)

        assert outputs["queue_position_logits"].shape == (batch_size, 5)
        assert outputs["score_margin"].shape == (batch_size, 1)
        assert outputs["cards_to_bar"].shape == (batch_size, 1)


class TestAsymmetricEncoder:
    """Test asymmetric transformer encoder."""

    def test_encoder_with_positional(self):
        """Test encoder with positional encoding (queue)."""
        encoder = AsymmetricTransformerEncoder(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_positional_encoding=True,
            max_len=10,
        )

        x = torch.randn(4, 5, 64)  # batch=4, seq=5, hidden=64
        output = encoder(x)

        assert output.shape == (4, 64), f"Expected (4, 64), got {output.shape}"

    def test_encoder_without_positional(self):
        """Test encoder without positional encoding (bar/hand)."""
        encoder = AsymmetricTransformerEncoder(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_positional_encoding=False,
            max_len=10,
        )

        x = torch.randn(4, 5, 64)
        output = encoder(x)

        assert output.shape == (4, 64)

    def test_encoder_with_padding_mask(self):
        """Test encoder handles padding mask correctly."""
        encoder = AsymmetricTransformerEncoder(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_positional_encoding=False,
        )

        x = torch.randn(4, 5, 64)
        # Mask last 2 positions
        padding_mask = torch.zeros(4, 5, dtype=torch.bool)
        padding_mask[:, 3:] = True

        output = encoder(x, padding_mask=padding_mask)
        assert output.shape == (4, 64)

    def test_encoder_all_padded(self):
        """Test encoder handles all-padded input."""
        encoder = AsymmetricTransformerEncoder(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        x = torch.randn(4, 5, 64)
        # All positions are padding
        padding_mask = torch.ones(4, 5, dtype=torch.bool)

        output = encoder(x, padding_mask=padding_mask)
        assert output.shape == (4, 64)


class TestNetworkV2SaveLoad:
    """Test model persistence."""

    def test_save_load_roundtrip(self, network):
        """Verify network can be saved and loaded correctly."""
        network.eval()

        torch.manual_seed(123)
        obs = torch.randn(2, OBSERVATION_DIM)

        with torch.no_grad():
            policy_orig, value_orig = network(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "network_v2.pt"
            torch.save(network.state_dict(), save_path)

            network_loaded = create_network_v2(network.config)
            network_loaded.load_state_dict(torch.load(save_path, weights_only=True))
            network_loaded.eval()

            with torch.no_grad():
                policy_loaded, value_loaded = network_loaded(obs)

        assert torch.allclose(policy_orig, policy_loaded, atol=1e-6), \
            "Loaded network produces different policy"
        assert torch.allclose(value_orig, value_loaded, atol=1e-6), \
            "Loaded network produces different value"


class TestNetworkV2WithRealState:
    """Test network with actual game states."""

    def test_forward_with_game_state(self, network):
        """Test forward pass with actual game observation."""
        game_state = state.initial_state(seed=42)
        obs_np = state_to_tensor(game_state, perspective=0)
        obs = torch.from_numpy(obs_np).unsqueeze(0)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert policy_logits.shape == (1, ACTION_DIM)
        assert value.shape == (1, 1)
        assert -1.0 <= value.item() <= 1.0


class TestConfigV2:
    """Test NetworkConfigV2 dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NetworkConfigV2()

        assert config.hidden_dim == 256
        assert config.queue_layers == 6
        assert config.bar_layers == 2
        assert config.use_dueling is True
        assert config.use_auxiliary_heads is True

    def test_config_to_dict(self):
        """Test config serialization."""
        config = NetworkConfigV2(hidden_dim=128, queue_layers=4)
        d = config.to_dict()

        assert d["hidden_dim"] == 128
        assert d["queue_layers"] == 4

    def test_config_from_dict(self):
        """Test config deserialization."""
        d = {"hidden_dim": 128, "queue_layers": 4, "use_dueling": False}
        config = NetworkConfigV2.from_dict(d)

        assert config.hidden_dim == 128
        assert config.queue_layers == 4
        assert config.use_dueling is False

    def test_config_immutable(self):
        """Test config is frozen."""
        config = NetworkConfigV2()

        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.hidden_dim = 512
