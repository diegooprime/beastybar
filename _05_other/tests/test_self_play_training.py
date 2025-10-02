"""Integration smoke tests for the self-play training scaffolding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from _01_simulator import action_space, observations, simulate, state
from _02_agents import FirstLegalAgent
from _03_training import encoders, models, policy_loader, ppo, rollout, self_play


def test_policy_loader_roundtrip(tmp_path: Path) -> None:
    observation_size = encoders.observation_size()
    action_size = len(action_space.canonical_actions())
    policy_config = models.PolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=(32,),
    )

    model = models.PolicyValueNet(policy_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    models.save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, step=42, metadata={"label": "test"})

    loader = policy_loader.load_policy(checkpoint=checkpoint_path, device="cpu")
    obs = observations.build_observation(state.initial_state(seed=7), perspective=0)
    logits = loader(obs)

    assert len(logits) == action_size
    assert all(isinstance(value, float) for value in logits)


def test_rollout_and_ppo_update() -> None:
    torch.manual_seed(0)
    observation_size = encoders.observation_size()
    action_size = len(action_space.canonical_actions())
    policy_config = models.PolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=(64,),
    )

    model = models.PolicyValueNet(policy_config)
    device = torch.device("cpu")
    model.to(device)

    rollout_config = rollout.RolloutConfig(min_steps=8, gamma=0.99, gae_lambda=0.95)
    batch = rollout.collect_rollouts(
        model=model,
        opponent_factories=[FirstLegalAgent],
        config=rollout_config,
        base_seed=123,
        device=device,
    )

    assert batch.steps >= 8
    assert batch.observations.shape[0] == batch.steps
    assert batch.action_masks.shape[0] == batch.steps

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = ppo.ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        config=ppo.PPOConfig(epochs=1, batch_size=4, clip_coef=0.2, value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5),
        device=device,
    )

    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_collect_rollouts_randomizes_starting_player(monkeypatch) -> None:
    torch.manual_seed(0)

    observation_size = encoders.observation_size()
    action_size = len(action_space.canonical_actions())
    policy_config = models.PolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=(32,),
    )

    model = models.PolicyValueNet(policy_config)
    device = torch.device("cpu")
    model.to(device)

    captured_starts: list[int] = []
    original_new_game = simulate.new_game

    def tracking_new_game(seed: int, *, starting_player: int = 0):
        captured_starts.append(starting_player)
        return original_new_game(seed, starting_player=starting_player)

    monkeypatch.setattr(simulate, "new_game", tracking_new_game)

    rollout_config = rollout.RolloutConfig(min_steps=32, gamma=0.99, gae_lambda=0.95)
    batch = rollout.collect_rollouts(
        model=model,
        opponent_factories=[FirstLegalAgent],
        config=rollout_config,
        base_seed=123,
        device=device,
    )

    assert batch.steps >= 32
    assert captured_starts, "collect_rollouts should start at least one episode"
    assert 0 in captured_starts
    assert 1 in captured_starts


def test_checkpoint_reservoir_produces_agents(tmp_path: Path) -> None:
    observation_size = encoders.observation_size()
    action_size = len(action_space.canonical_actions())
    policy_config = models.PolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=(32,),
    )

    model = models.PolicyValueNet(policy_config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    models.save_checkpoint(checkpoint_path, model=model, optimizer=None, step=10)

    reservoir = self_play.CheckpointReservoir(max_size=2, device=torch.device("cpu"))
    reservoir.add_checkpoint(checkpoint_path)
    reservoir.add_checkpoint(checkpoint_path)

    assert len(reservoir) == 1
    assert reservoir.paths == [checkpoint_path.resolve()]

    factories = reservoir.factories
    assert len(factories) == 1
    agent = factories[0]()

    base_state = simulate.new_game(seed=5, starting_player=0)
    opponent_state = state.set_active_player(base_state, 1)
    opponent_view = state.mask_state_for_player(opponent_state, 1)
    agent.start_game(opponent_view)
    legal = simulate.legal_actions(opponent_state, 1)
    assert legal
    chosen = agent.select_action(opponent_state, legal)
    assert chosen in legal

    second_checkpoint = tmp_path / "second.pt"
    models.save_checkpoint(second_checkpoint, model=model, optimizer=None, step=20)
    third_checkpoint = tmp_path / "third.pt"
    models.save_checkpoint(third_checkpoint, model=model, optimizer=None, step=30)

    reservoir.add_checkpoint(second_checkpoint)
    reservoir.add_checkpoint(third_checkpoint)

    assert len(reservoir) == 2
    assert reservoir.paths == [second_checkpoint.resolve(), third_checkpoint.resolve()]


def test_rolling_metric_aggregator_outputs(tmp_path: Path) -> None:
    aggregator = self_play.RollingMetricAggregator(directory=tmp_path, window_size=3)
    aggregator.bootstrap()

    for idx in range(4):
        payload = {
            "timestamp": f"2024-01-0{idx + 1}T00:00:00+00:00",
            "iteration": idx + 1,
            "steps": (idx + 1) * 100,
            "episodes": (idx + 1) * 5,
            "rolloutSteps": 64 + idx,
            "batchEpisodes": 4 + idx,
            "avgEpisodeReward": 0.1 * (idx + 1),
            "meanEntropy": 0.2 + 0.01 * idx,
            "policy_loss": -0.05 * (idx + 1),
            "value_loss": 0.5 + 0.05 * idx,
            "entropy": 0.1 + 0.02 * idx,
            "approx_kl": 0.01 + 0.001 * idx,
            "clip_fraction": 0.02 + 0.002 * idx,
        }
        aggregator.record(payload)

    json_path = tmp_path / "rolling_metrics.json"
    assert json_path.exists()
    history = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(history) == 4

    last_entry = history[-1]
    assert last_entry["window"] == 3
    assert last_entry["steps"] == 400
    assert last_entry["avgEpisodeReward_mean"] == pytest.approx((0.2 + 0.3 + 0.4) / 3)
    assert last_entry["policy_loss_min"] == pytest.approx(-0.2)
    assert last_entry["value_loss_max"] == pytest.approx(0.65)

    parquet_path = tmp_path / "rolling_metrics.parquet"
    if parquet_path.exists():
        import pyarrow.parquet as pq  # type: ignore[import]

        table = pq.read_table(parquet_path)
        assert table.num_rows == 4
        last_row = table.to_pylist()[-1]
        assert last_row["window"] == 3
        assert last_row["avgEpisodeReward_mean"] == pytest.approx((0.2 + 0.3 + 0.4) / 3)


def test_eval_knobs_round_trip_config(tmp_path: Path) -> None:
    config_path = tmp_path / "self_play.json"
    config_payload = {"eval_games": 42, "eval_seed": 9876}
    config_path.write_text(json.dumps(config_payload) + "\n", encoding="utf-8")

    args = self_play._parse_args(["--config", str(config_path)])
    base_config = self_play._load_config(args.config)
    resolved = self_play._merge_config(base_config, args)

    assert resolved.eval_games == 42
    assert resolved.eval_seed == 9876

    expected_min_games = len(resolved.opponents) * resolved.eval_games
    assert resolved.promotion_min_games == expected_min_games


def test_run_evaluation_suite_writes_summary(tmp_path: Path) -> None:
    observation_size = encoders.observation_size()
    action_size = len(action_space.canonical_actions())
    policy_config = models.PolicyConfig(
        observation_size=observation_size,
        action_size=action_size,
        hidden_sizes=(32,),
    )

    model = models.PolicyValueNet(policy_config)

    checkpoints_dir = tmp_path / "checkpoints"
    metrics_dir = tmp_path / "metrics"
    rollouts_dir = tmp_path / "rollouts"
    eval_dir = tmp_path / "eval"
    for directory in (checkpoints_dir, metrics_dir, rollouts_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "candidate.pt"
    models.save_checkpoint(checkpoint_path, model=model, optimizer=None, step=12)

    config = self_play.TrainingConfig(
        phase="p3",
        seed=123,
        opponents=["first"],
        total_steps=10,
        eval_frequency=5,
        artifact_root=tmp_path,
        run_id="test-run",
        resume_from=None,
        notes=None,
        rollout_steps=8,
        reservoir_size=2,
        eval_games=2,
        eval_seed=321,
        promotion_min_games=2,
        promotion_min_win_rate=0.0,
        promotion_min_elo_delta=-100.0,
        gamma=0.99,
        gae_lambda=0.95,
        margin_weight=0.25,
        jitter_scale=0.01,
        learning_rate=3e-4,
        ppo_epochs=1,
        ppo_batch_size=4,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        device="cpu",
    )

    artifacts = self_play.RunArtifacts(
        root=tmp_path,
        checkpoints=checkpoints_dir,
        metrics=metrics_dir,
        rollouts=rollouts_dir,
        eval=eval_dir,
        manifest=tmp_path / "manifest.json",
        champion_manifest=tmp_path / "champion.json",
    )

    eval_summary = self_play._run_evaluation_suite(
        checkpoint=checkpoint_path,
        config=config,
        artifacts=artifacts,
        iteration=1,
        step=12,
        eval_device=torch.device("cpu"),
    )

    assert eval_summary is not None
    assert eval_summary.path.exists()
    assert eval_summary.games == config.eval_games
    assert eval_summary.win_rate >= 0.0

    payload = json.loads(eval_summary.path.read_text(encoding="utf-8"))
    assert payload["step"] == 12
    assert payload["gamesPerOpponent"] == config.eval_games
    assert payload["opponents"]
    opponent_entry = payload["opponents"][0]
    assert opponent_entry["opponent"] == "first"
    assert opponent_entry["games"] == config.eval_games

    artifacts.manifest.write_text(json.dumps({"runId": config.run_id}) + "\n", encoding="utf-8")

    promoted, champion = self_play._maybe_promote_checkpoint(
        checkpoint=checkpoint_path,
        summary=eval_summary,
        config=config,
        artifacts=artifacts,
        champion=None,
    )

    assert promoted is True
    assert champion is not None
    assert artifacts.champion_manifest.exists()

    champion_payload = json.loads(artifacts.champion_manifest.read_text(encoding="utf-8"))
    assert champion_payload["model"] == "_03_training.policy_loader:load_policy"
    assert champion_payload["checkpoint"].endswith("candidate.pt")
    assert champion_payload["evaluation"]["winRate"] == pytest.approx(eval_summary.win_rate)

    manifest_payload = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest_payload["champion"]["checkpoint"].endswith("candidate.pt")
    assert manifest_payload["champion"]["games"] == eval_summary.games
