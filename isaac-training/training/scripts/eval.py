"""
Closed-loop NavRL evaluation with optional depth-image noise injection.

Runs the policy in Isaac Sim for one full episode length per noise condition,
logs all results to wandb, and saves a video per condition.

Usage
-----
  # Clean evaluation only:
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt

  # Clean + all noise conditions defined in cfg.eval.noise_conditions:
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt \
      eval.run_noise=true

  # Single noise condition override:
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt \
      eval.run_noise=true \
      "eval.noise_conditions=[{name: gaussian_high, type: gaussian, sigma: 0.10}]"

  # Only run a subset of noise types (disable the rest without editing the
  # noise_conditions list):
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint=wandb/run-X/files/checkpoint_5000.pt \
      eval.run_noise=true \
      "eval.types=[gaussian, dropout]"
"""
import os
import gc
import statistics
import datetime
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import isaacsim  # must come before omni.isaac.kit on Isaac Sim 4.x
from omni.isaac.kit import SimulationApp
# These must be imported before SimulationApp() is constructed (matches
# train.py's import order) — importing ppo/tensordict AFTER SimulationApp
# starts collides with Isaac Sim's bundled functorch build and crashes with
# "undefined symbol: _ZN2at9functorch11addBatchDimERKNS_6TensorEll".
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils import evaluate, make_batched_gru_primer

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="eval_noise", version_base=None)
def main(cfg: DictConfig):
    # ── Simulation App ────────────────────────────────────────────────────────
    # anti_aliasing=0 (disabled) by default — this is a training/eval speed
    # path, not a presentation one; the depth sensor uses a separate render
    # product unaffected by this setting. Override anti_aliasing=<1..4> (see
    # SimulationApp docstring: 1=TAA, 2=FXAA, 3=DLSS, 4=RTXAA) for a cleaner
    # one-off recording — RTX's real-time renderer relies on temporal
    # accumulation/denoising that AA=0 skips, which is the main source of the
    # grainy look in eval_video=true recordings.
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": int(getattr(cfg, "anti_aliasing", 0))})

    # ── Wandb ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        mode=cfg.wandb.mode,
    )

    # ── Environment ───────────────────────────────────────────────────────────
    _env_module = __import__(getattr(cfg, "env_script", "env_depth"))
    env = _env_module.NavigationEnv(cfg)

    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
    transformed_env.set_seed(cfg.seed)

    # ── Policy ────────────────────────────────────────────────────────────────
    policy = PPO(cfg.algo, transformed_env.observation_spec,
                 transformed_env.action_spec, cfg.device)

    if not getattr(cfg, "checkpoint", None):
        raise ValueError("Set checkpoint=<path> on the command line.")

    state_dict = torch.load(cfg.checkpoint, map_location=cfg.device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[eval] Skipped keys (training-only): {unexpected[:4]}"
              + (" …" if len(unexpected) > 4 else ""))
    if missing:
        print(f"[eval] MISSING keys (left at random init!): {missing[:8]}"
              + (" …" if len(missing) > 8 else ""))
    print(f"[eval] Loaded checkpoint: {cfg.checkpoint}")

    # network_type: gru needs the env to carry an "is_init" reset flag and a
    # "recurrent_state" hidden-state key across steps -- must be wired before
    # the warm-up rollout below, the first reset/step ever run on this env.
    # No-op for cnn/transformer.
    if getattr(cfg.algo, "network_type", "cnn") == "gru":
        transformed_env.append_transform(InitTracker())
        transformed_env.append_transform(
            make_batched_gru_primer(policy.gru_module, cfg.env.num_envs, cfg.device)
        )
        # Critic's independent GRU (see ppo.py's asymmetric actor/critic
        # extractor) has its own hidden-state key ("critic_recurrent_state")
        # -- needs its own primer alongside the actor's above.
        transformed_env.append_transform(
            make_batched_gru_primer(policy.critic_gru_module, cfg.env.num_envs, cfg.device)
        )

    # PPO overrides train() for RL — a freshly constructed policy defaults to
    # nn.Module's training=True and is never toggled, so any BatchNorm in the
    # (possibly frozen) encoder uses live batch statistics instead of the
    # trained running stats (freeze_encoder only sets requires_grad=False, it
    # never calls .eval()) — silently degrades inference, worst for frozen
    # pretrained encoders. Call eval() on submodules directly, matching what
    # evaluation/eval_noise_robustness.py already does for the same reason.
    for m in [policy.feature_extractor, policy.critic_feature_extractor, policy.actor, policy.critic]:
        m.eval()

    # ── Warm-up rollout ───────────────────────────────────────────────────────
    # train.py's periodic eval always runs after the SyncDataCollector has
    # already reset/stepped the env at least once (gathering its first
    # training batch). Calling evaluate() as the very first reset ever done on
    # a freshly-built env (this script's flow) hits PhysX "illegal to call
    # setGlobalPose() while eENABLE_DIRECT_GPU_API is enabled" errors on the
    # first reset of every env. Confirmed harmless/self-healing (0 errors on
    # every subsequent reset, correct reach_goal once the real bug below was
    # fixed) — this warm-up just moves the one-time error here instead of
    # having it spam the real "clean" condition's log. Deliberately NOT
    # enabling render here (env.enable_render(True)) to avoid allocating any
    # RTX viewport/render-product resources beyond what eval_video actually
    # needs.
    print("[eval] Warm-up rollout (priming physics GPU pipeline before real eval) …")
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        transformed_env.rollout(max_steps=2, policy=policy, auto_reset=True,
                                 break_when_any_done=False)
    transformed_env.reset()

    # ── Build noise conditions list ───────────────────────────────────────────
    # Always run clean first; append configured noise conditions if enabled.
    # `eval.types`, if set, filters noise_conditions down to the listed types
    # (e.g. types=[gaussian, dropout] skips all cutout/quantization entries).
    # `eval.include_clean=false` skips the auto-prepended clean condition —
    # needed to run exactly one condition per process: a 350-env x 2200-step
    # rollout is already tens of GB, and running two of them back-to-back in
    # the same process reliably OOMs a 31GB GPU no matter how aggressively
    # the first one's buffers are freed (gc.collect + synchronize + no_grad
    # all tried, none recovered it) — so the driver script runs one condition
    # per subprocess instead, each with a guaranteed-fresh CUDA state.
    # `eval.include_maxfill` (default true): the dropout/cutout "*_maxfill"
    # conditions duplicate the zero-fill ones with fill_value=1.0 (see
    # eval_noise.yaml comment) — set false to skip them and only run the
    # zero-fill originals, halving dropout/cutout runtime. Symmetrically,
    # `eval.include_zerofill` (default true) lets a validation probe run
    # maxfill-only without re-running the already-validated zero-fill half.
    include_maxfill  = bool(cfg.eval.get("include_maxfill", True))
    include_zerofill = bool(cfg.eval.get("include_zerofill", True))
    conditions = [{"name": "clean", "type": None}] if cfg.eval.get("include_clean", True) else []
    if cfg.eval.get("run_noise", True):
        enabled_types = cfg.eval.get("types", None)
        for nc in cfg.eval.get("noise_conditions", []):
            if enabled_types is not None and nc.get("type") not in enabled_types:
                continue
            # zerofill/maxfill only applies to dropout/cutout (the only types
            # with a fill_value axis) — gaussian/quantization always run.
            if nc.get("type") in ("dropout", "cutout"):
                is_maxfill = nc.get("name", "").endswith("_maxfill")
                if is_maxfill and not include_maxfill:
                    continue
                if not is_maxfill and not include_zerofill:
                    continue
            conditions.append(OmegaConf.to_container(nc, resolve=True))

    # ── Evaluate each condition ───────────────────────────────────────────────
    # `eval.n_trials` (default 1): repeat each NOISY condition this many times
    # with independent noise draws (different torch.manual_seed per trial —
    # see evaluate()'s env.set_seed(seed) call), all within this same process
    # so the underlying scenario (obstacle terrain, deterministic eval-mode
    # spawn/goal positions) is held fixed across trials — only the sampled
    # noise (gaussian values / dropout mask / cutout patch position) differs.
    # "clean" has no randomness, so it always runs exactly once regardless.
    # Logs per-trial results plus the aggregate mean (under the original key
    # name, for backward compat with single-trial logs) and, when
    # n_trials > 1, a "<key>_std" alongside it.
    n_trials = int(cfg.eval.get("n_trials", 1))

    for cond in conditions:
        name      = cond["name"]
        noise_cfg = {k: v for k, v in cond.items() if k != "name"} or None
        trials    = n_trials if noise_cfg is not None else 1

        trial_infos = []
        for t in range(trials):
            trial_seed = cfg.seed * 1000 + t if trials > 1 else cfg.seed
            suffix = f" (trial {t+1}/{trials}, seed={trial_seed})" if trials > 1 else ""
            print(f"\n[eval] === condition: {name}{suffix} ===")

            env.eval()
            eval_info = evaluate(
                env=transformed_env,
                policy=policy,
                cfg=cfg,
                seed=trial_seed,
                exploration_type=ExplorationType.MEAN,
                noise_cfg=noise_cfg,
            )
            env.train()
            env.reset()
            # A single 350-env x 2200-step depth rollout is already ~25GB; without
            # a hard gc.collect() before empty_cache(), lingering Python refs
            # (e.g. inside the just-returned eval_info closure / TensorDict
            # internals) keep the CUDA allocator from reclaiming it, and the next
            # condition's rollout OOMs on a 31GB GPU.
            gc.collect()
            torch.cuda.empty_cache()

            trial_infos.append(eval_info)
            if trials > 1:
                run.log({f"{name}/trial{t}/{k}": v for k, v in eval_info.items()})

        # Aggregate: mean under the original key (backward-compatible with
        # single-trial logs), plus "_std" alongside it when trials > 1.
        keys = [k for k in trial_infos[0] if isinstance(trial_infos[0][k], float)]
        agg = {}
        for k in keys:
            vals = [ti[k] for ti in trial_infos]
            agg[k] = statistics.mean(vals)
            if trials > 1:
                agg[f"{k}_std"] = statistics.pstdev(vals)

        prefixed = {f"{name}/{k}": v for k, v in agg.items()}
        # "recording" (a wandb.Video, when eval_video=true) isn't a float, so
        # the filter above drops it — averaging video objects across trials
        # makes no sense anyway, so just pass through the first trial's.
        if "recording" in trial_infos[0]:
            prefixed[f"{name}/recording"] = trial_infos[0]["recording"]
        run.log(prefixed)
        print(f"[eval] {name}  "
              + "  ".join(f"{k.split('.')[-1]}={v:.3f}"
                          for k, v in agg.items()
                          if not k.endswith("_std")))

    # ── Done ──────────────────────────────────────────────────────────────────
    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
