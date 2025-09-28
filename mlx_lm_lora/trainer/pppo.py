from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Dict, List, Tuple

from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import iterate_batches

###################################################################################################
# Utility: gradient checkpointing (unchanged)
###################################################################################################

def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


###################################################################################################
# Φ-PPO: RL training from scratch on SFT data
################################################################################################### RL training from scratch on SFT data
###################################################################################################

@dataclass
class PPPOTrainingArgs:
    batch_size: int = 2
    iters: int = 100
    rollout_max_len: int = 128
    steps_per_eval: int = 200
    steps_per_report: int = 10
    steps_per_save: int = 200
    max_seq_length: int = 2048
    adapter_file: str = "adapters.safetensors"
    grad_checkpoint: bool = False

    # PPO
    ppo_epochs: int = 2
    ppo_minibatch_size: int = 256  # per-token samples in advantage buffer
    clip_eps: float = 0.15
    entropy_coef: float = 0.02
    kl_mu_coef: float = 0.10
    kl_old_target: float = 0.05  # nats per token (approx)

    # Advantage/returns
    gamma: float = 1.0
    lam: float = 0.9
    advantage_clip: float = 5.0

    # Potential shaping
    len_penalty: float = 0.1
    max_f1: float = 1.0

    # Decoding
    temperature: float = 1.0
    top_p: float = 0.95


class SoftF1Potential:
    """Streaming token-level soft-F1 between a generated prefix and reference y*.
    Uses multiset 1-gram overlap with a length penalty to keep Φ in [0,1].
    """
    def __init__(self, y_star: np.ndarray, len_penalty: float = 0.1, eps: float = 1e-6):
        self.eps = eps
        self.len_penalty = len_penalty
        self.y_star = y_star.tolist()
        self.ref_counts: Dict[int, int] = {}
        for t in self.y_star:
            self.ref_counts[t] = self.ref_counts.get(t, 0) + 1
        self.gen_counts: Dict[int, int] = {}
        self.overlap = 0
        self.t = 0
        self.ref_len = max(1, len(self.y_star))
        self.phi = 0.0

    def _f1(self) -> float:
        if self.t == 0:
            return 0.0
        precision = self.overlap / (self.t + self.eps)
        recall = self.overlap / (self.ref_len + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        # smooth length penalty around ratio 1.0
        ratio = self.t / self.ref_len
        penalty = self.len_penalty * abs(ratio - 1.0)
        return max(0.0, min(1.0, f1 - penalty))

    def update(self, tok: int) -> float:
        prev = self.phi
        # update counts & overlap
        self.gen_counts[tok] = self.gen_counts.get(tok, 0) + 1
        if self.gen_counts[tok] <= self.ref_counts.get(tok, 0):
            self.overlap += 1
        self.t += 1
        self.phi = self._f1()
        return self.phi - prev


def _extract_prompt_and_ref(tokens: np.ndarray, offset: int, total_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a concatenated sequence (prompt + target) using (offset, total_len).
    offset: first supervised index (1-indexed in original mask); here we use as 1-based step index
    total_len: effective length of sequence (<= max_seq_length)
    Returns (prompt_ids, y_star_ids)
    """
    # In the original SFT mask, steps start at 1 for targets. We treat `offset` as the first target step (1-based),
    # so the prompt ends at index `offset-1` in step space. Our `tokens` is 0-based.
    prompt_end = max(0, int(offset) - 1)  # number of steps before first supervised token
    prompt_ids = tokens[:prompt_end]
    y_star_ids = tokens[prompt_end:int(total_len)]
    return prompt_ids.astype(np.int32), y_star_ids.astype(np.int32)


def _log_softmax(x: mx.array) -> mx.array:
    return x - mx.log(mx.sum(mx.exp(x), axis=-1, keepdims=True))


def _entropy_from_logits(logits_last: mx.array) -> mx.array:
    # logits_last: [N, V]
    logp = _log_softmax(logits_last)
    p = mx.exp(logp)
    return -mx.sum(p * logp, axis=-1)  # [N]


def _sample_from_logits(logits_last: mx.array, temperature: float = 1.0, top_p: float = 0.95) -> Tuple[np.ndarray, mx.array]:
    # logits_last: [B, V]
    logits = logits_last / max(1e-6, temperature)
    # nucleus sampling
    sorted_logits = mx.sort(logits, axis=-1, descending=True)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cdf = mx.cumsum(sorted_probs, axis=-1)
    # mask tokens beyond nucleus
    mask = cdf > top_p
    # shift mask right to always keep first token
    mask = mx.concatenate([mx.zeros((*mask.shape[:-1], 1)), mask[..., :-1]], axis=-1)
    # map back to original ordering
    # For simplicity in MLX without argsort-gather, we fall back to numpy sampling over full softmax
    probs = mx.softmax(logits, axis=-1)
    probs_np = np.asarray(probs)
    actions = []
    for i in range(probs_np.shape[0]):
        p = probs_np[i]
        # enforce top-p by zeroing smallest tail until cumulative>top_p
        order = np.argsort(-p)
        c = np.cumsum(p[order])
        k = np.searchsorted(c, top_p) + 1
        keep = order[:k]
        p_masked = np.zeros_like(p)
        p_masked[keep] = p[keep]
        if p_masked.sum() <= 0:
            p_masked = p
        p_masked = p_masked / p_masked.sum()
        a = np.random.choice(len(p_masked), p=p_masked)
        actions.append(a)
    actions = np.array(actions, dtype=np.int64)
    logp = nn.log_softmax(logits, axis=-1)
    chosen_logp = mx.take_along_axis(logp, mx.array(actions)[:, None], axis=-1).squeeze(-1)
    return actions, chosen_logp


def build_behavior_prior(dataset, iterate_batches: Callable, vocab_size: int, max_seq_length: int, batch_size: int) -> mx.array:
    """Estimate a unigram prior μ over the vocabulary from the dataset.
    Returns μ as a probability vector [V].
    """
    counts = np.zeros(vocab_size, dtype=np.float64)
    total = 0
    for (batch, lengths) in iterate_batches(dataset, batch_size=batch_size, max_seq_length=max_seq_length, train=False):
        b = np.asarray(batch)
        for i in range(b.shape[0]):
            L = int(lengths[i, 1])
            toks = b[i, :L]
            uniq, cnt = np.unique(toks, return_counts=True)
            counts[uniq] += cnt
            total += L
    counts = counts + 1.0  # Laplace smoothing
    mu = counts / counts.sum()
    return mx.array(mu.astype(np.float32))


class AdvantageBuffer:
    def __init__(self):
        self.contexts: List[np.ndarray] = []   # [L_ctx]
        self.actions: List[int] = []
        self.old_logprobs: List[float] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def add(self, ctx: np.ndarray, a: int, logp: float, adv: float, ret: float):
        self.contexts.append(ctx.astype(np.int32))
        self.actions.append(int(a))
        self.old_logprobs.append(float(logp))
        self.advantages.append(float(adv))
        self.returns.append(float(ret))

    def to_batches(self, minibatch_size: int):
        N = len(self.actions)
        idx = np.random.permutation(N)
        for i in range(0, N, minibatch_size):
            j = idx[i:i+minibatch_size]
            ctxs = self._pad([self.contexts[k] for k in j])  # [n, L]
            acts = mx.array([self.actions[k] for k in j], dtype=mx.int32)
            old_lp = mx.array([self.old_logprobs[k] for k in j], dtype=mx.float32)
            adv = mx.array([self.advantages[k] for k in j], dtype=mx.float32)
            ret = mx.array([self.returns[k] for k in j], dtype=mx.float32)
            yield ctxs, acts, old_lp, adv, ret

    @staticmethod
    def _pad(batch_ctx: List[np.ndarray]) -> mx.array:
        L = max(len(c) for c in batch_ctx)
        arr = np.zeros((len(batch_ctx), L), dtype=np.int32)
        for i, c in enumerate(batch_ctx):
            arr[i, :len(c)] = c
        return mx.array(arr)


def _compute_logprob_for_last_token(model, contexts: mx.array, actions: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """Run model on contexts and return (logprob(a|ctx), entropy, logits_last).
    contexts: [B, L] token ids
    actions: [B]
    """
    logits = model(contexts)  # [B, L, V]
    last_logits = logits[:, -1, :]
    logp_all = nn.log_softmax(last_logits, axis=-1)
    chosen_lp = mx.take_along_axis(logp_all, actions[:, None], axis=-1).squeeze(-1)
    ent = -mx.sum(mx.exp(logp_all) * logp_all, axis=-1)
    return chosen_lp, ent, last_logits


def evaluate_phi(
    model,
    dataset,
    batch_size,
    num_batches,
    iterate_batches: Callable,
    args: PPPOTrainingArgs,
    max_seq_length=2048,
):
    """Generate with current policy and report mean terminal Φ (soft-F1) on val set."""
    model.eval()
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    total_F = 0.0
    n = 0
    for _, (batch, lengths) in zip(index_iterator, iterate_batches(dataset, batch_size, max_seq_length, train=False)):
        b = np.asarray(batch)
        len_np = np.asarray(lengths)
        B = b.shape[0]
        for i in range(B):
            offset, L = int(len_np[i, 0]), int(len_np[i, 1])
            prompt, y_star = _extract_prompt_and_ref(b[i, :L], offset, L)
            pot = SoftF1Potential(y_star, len_penalty=args.len_penalty)
            ctx = prompt.copy()
            # rollout up to length of reference for fair comparison
            for _ in range(min(args.rollout_max_len, max(1, len(y_star)))):
                inp = mx.array(ctx[None, :])
                logits = model(inp)
                last = logits[:, -1, :]
                action_ids, _ = _sample_from_logits(last, temperature=args.temperature, top_p=args.top_p)
                a = int(action_ids[0])
                pot.update(a)
                ctx = np.concatenate([ctx, np.array([a], dtype=np.int32)], axis=0)
            total_F += pot.phi
            n += 1
    model.train()
    return total_F / max(1, n)


def train_phi_ppo(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    iterate_batches: Callable = iterate_batches,
    args: PPPOTrainingArgs = PPPOTrainingArgs(),
    training_callback: TrainingCallback = None,
):
    """
    Φ-PPO: RL from scratch on SFT data using potential-shaped dense rewards.
    - Uses unigram behavior prior μ for KL regularization.
    - PPO with clipped ratios on per-token samples.
    - Simple baseline: batch-mean returns (critic-less). Can be extended to a critic.
    """
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    world = mx.distributed.init()
    rank = world.rank()

    if args.grad_checkpoint and hasattr(model, "layers") and len(model.layers) > 0:
        grad_checkpoint(model.layers[0])

    # infer vocab size from a dummy forward
    # NOTE: assumes model returns logits [B, T, V]
    dummy = mx.array(np.zeros((1, 2), dtype=np.int32))
    V = int(model(dummy).shape[-1])
    mu = build_behavior_prior(train_dataset, iterate_batches, vocab_size=V, max_seq_length=args.max_seq_length, batch_size=args.batch_size)

    def kl_to_mu_from_logits(last_logits: mx.array) -> mx.array:
        # KL(pi||mu) averaged over batch; mu: [V]
        logp = nn.log_softmax(last_logits, axis=-1)
        p = mx.exp(logp)
        log_mu = mx.log(mu)[None, :]
        kl = mx.sum(p * (logp - log_mu), axis=-1)  # [B]
        return kl

    # Main loop
    pbar = tqdm(range(1, args.iters + 1), desc="Φ-PPO Training", disable=rank != 0)
    trained_tokens = 0
    last_eval = None

    for it in pbar:
        # === Build on-policy rollouts for a batch of prompts ===
        batch, lengths = next(iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ))
        b = np.asarray(batch)
        len_np = np.asarray(lengths)

        advbuf = AdvantageBuffer()
        total_tokens_this_iter = 0

        for i in range(b.shape[0]):
            offset, L = int(len_np[i, 0]), int(len_np[i, 1])
            prompt, y_star = _extract_prompt_and_ref(b[i, :L], offset, L)
            pot = SoftF1Potential(y_star, len_penalty=args.len_penalty)

            ctx = prompt.copy()
            rewards: List[float] = []
            actions: List[int] = []
            logps: List[float] = []

            # rollout up to min(rollout_max_len, len(y*)) to compare fairly
            H = min(args.rollout_max_len, max(1, len(y_star)))
            for t in range(H):
                inp = mx.array(ctx[None, :])
                logits = model(inp)
                last = logits[:, -1, :]
                action_ids, chosen_logp = _sample_from_logits(last, temperature=args.temperature, top_p=args.top_p)
                a = int(action_ids[0])
                r_t = pot.update(a)
                ctx = np.concatenate([ctx, np.array([a], dtype=np.int32)], axis=0)

                rewards.append(r_t)
                actions.append(a)
                logps.append(float(chosen_logp.item()))

            # returns & baseline (critic-less): use reward-to-go and center
            rets = np.array(rewards, dtype=np.float32)
            rets = np.flip(np.cumsum(np.flip(rets)))  # reward-to-go
            adv = rets - rets.mean()
            adv = np.clip(adv, -args.advantage_clip, args.advantage_clip)

            # add per-time-step samples
            # context for step t is prompt + generated[:t]
            ctx_base = prompt.copy()
            for t in range(len(actions)):
                advbuf.add(ctx_base.copy(), actions[t], logps[t], adv[t], rets[t])
                ctx_base = np.concatenate([ctx_base, np.array([actions[t]], dtype=np.int32)], axis=0)

            total_tokens_this_iter += len(actions)

        trained_tokens += total_tokens_this_iter

        # === PPO updates over advantage buffer ===
        def ppo_minibatch_loss(model, contexts, actions, old_logp, advantages, returns):
            new_logp, ent, last_logits = _compute_logprob_for_last_token(model, contexts, actions)
            ratio = mx.exp(new_logp - old_logp)
            unclipped = ratio * advantages
            clipped = mx.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
            policy_loss = -mx.mean(mx.minimum(unclipped, clipped))

            entropy_loss = -mx.mean(ent)  # we will multiply by entropy_coef
            # Approximate KL(new||old) for reporting (not penalized directly):
            approx_kl = mx.mean(old_logp - new_logp)

            # KL to behavior prior μ
            kl_mu = mx.mean(kl_to_mu_from_logits(last_logits))

            total_loss = policy_loss + args.entropy_coef * entropy_loss + args.kl_mu_coef * kl_mu
            return total_loss, policy_loss, entropy_loss, kl_mu, approx_kl

        loss_and_grad = nn.value_and_grad(model, ppo_minibatch_loss)

        # Multiple PPO epochs over shuffled minibatches
        policy_losses = []
        ent_losses = []
        kl_mus = []
        approx_kls = []

        for _ in range(args.ppo_epochs):
            for ctxs, acts, old_lp, adv, ret in advbuf.to_batches(args.ppo_minibatch_size):
                (tot, pol, entl, klm, akl), grad = loss_and_grad(model, ctxs, acts, old_lp, adv, ret)
                optimizer.update(model, grad)
                # Accumulate stats
                policy_losses.append(pol.item())
                ent_losses.append(entl.item())
                kl_mus.append(klm.item())
                approx_kls.append(akl.item())

        # === Reporting & evaluation ===
        if it % args.steps_per_report == 0 or it == args.iters:
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: tokens {trained_tokens}, "
                    f"PPO/policy {np.mean(policy_losses):.4f}, "
                    f"entropy {np.mean(ent_losses):.4f}, "
                    f"KL(mu) {np.mean(kl_mus):.4f}, "
                    f"KL(old)~ {np.mean(approx_kls):.4f}"
                )
                pbar.set_postfix({
                    'pol': f"{np.mean(policy_losses):.3f}",
                    'KL~': f"{np.mean(approx_kls):.3f}",
                })

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "trained_tokens": trained_tokens,
                    "policy_loss": float(np.mean(policy_losses)),
                    "entropy": float(np.mean(ent_losses)),
                    "kl_mu": float(np.mean(kl_mus)),
                    "approx_kl_old": float(np.mean(approx_kls)),
                }
                training_callback.on_train_loss_report(train_info)

        if it % args.steps_per_eval == 0 or it == args.iters:
            tic = time.perf_counter()
            val_phi = evaluate_phi(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=10,  # quick probe; adjust as needed
                iterate_batches=iterate_batches,
                args=args,
                max_seq_length=args.max_seq_length,
            )
            took = time.perf_counter() - tic
            last_eval = val_phi
            if rank == 0:
                tqdm.write(f"Iter {it}: Val Φ (soft-F1) {val_phi:.4f} (took {took:.2f}s)")
            if training_callback is not None:
                training_callback.on_val_loss_report({"iteration": it, "val_phi": float(val_phi), "val_time": took})

        if it % args.steps_per_save == 0 or it == args.iters:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            if rank == 0:
                tqdm.write(
                    f"Saved adapter weights to {args.adapter_file} and {checkpoint}. "
                    f"Last Val Φ={last_eval if last_eval is not None else float('nan'):.4f}"
                )

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        tqdm.write(f"Saved final weights to {args.adapter_file}.")


###################################################################################################
# Optional: Advantage-Weighted Warm-start (offline RL on gold-path)
###################################################################################################

def advantage_weighted_warmstart(
    model,
    optimizer,
    dataset,
    args: PPPOTrainingArgs,
    epochs: int = 1,
    beta: float = 4.0,
    w_max: float = 5.0,
    iterate_batches: Callable = iterate_batches,
):
    """Offline RL on gold trajectories: weight CE by ΔΦ token-advantages.
    Keeps the API close to SFT but uses informative weights.
    """
    model.train()

    def weighted_ce_loss(model, batch, lengths):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        # Build mask region and compute ΔΦ per token on the gold path
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

        # Compute per-token weights on CPU/Numpy for simplicity
        b = np.asarray(batch)
        len_np = np.asarray(lengths)
        W = np.ones_like(b[:, 1:], dtype=np.float32)  # align with targets positions
        for i in range(b.shape[0]):
            offset, L = int(len_np[i, 0]), int(len_np[i, 1])
            prompt, y_star = _extract_prompt_and_ref(b[i, :L], offset, L)
            pot = SoftF1Potential(y_star, len_penalty=args.len_penalty)
            # walk along gold tokens to get ΔΦ
            deltas = []
            for tok in y_star:
                deltas.append(pot.update(int(tok)))
            deltas = np.array(deltas, dtype=np.float32)
            w = np.exp(beta * deltas)
            w = np.clip(w, 0.0, w_max)
            # place into the aligned stripe of W corresponding to supervised region
            start = max(0, offset - 1)  # targets indexing
            end = min(W.shape[1], start + len(deltas))
            W[i, start:end] = w[: end - start]
        W = mx.array(W)

        ce = nn.losses.cross_entropy(logits, targets)
        loss = (ce * W * mask).sum() / (mask.sum() + 1e-6)
        ntoks = mask.sum()
        return loss, ntoks

    loss_and_grad = nn.value_and_grad(model, weighted_ce_loss)

    for ep in range(epochs):
        pbar = tqdm(range(1, 1 + 100), desc=f"AWT Warmstart ep{ep+1}")
        for _ in pbar:
            batch, lengths = next(iterate_batches(dataset, batch_size=args.batch_size, max_seq_length=args.max_seq_length, train=True))
            (loss, ntoks), grad = loss_and_grad(model, batch, lengths)
            optimizer.update(model, grad)
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
