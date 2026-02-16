# Technical Documentation

## TranAD Architecture

### Overview

TranAD is a transformer-based encoder-decoder model for unsupervised anomaly detection in multivariate time series. The model learns to reconstruct windows of normal data during training. At inference time, the reconstruction error between the model's output and the actual input serves as the anomaly score: windows the model cannot reconstruct well are likely anomalous.

The implementation in `app/tranad_model.py` follows the architecture described in the original paper (Tuli et al., VLDB 2022) but is primarily faithful to the authors' reference code at `imperial-qore/TranAD`, which diverges from the paper in several notable ways. Where these divergences exist, both variants are implemented and selectable via `TranADConfig`.

### Configuration

All architecture and training parameters are centralized in the `TranADConfig` dataclass. The key architectural fields are:

- `n_features`: number of input dimensions (38 for SMD data)
- `window_size`: sliding window length (default 10)
- `n_heads`: number of attention heads (set equal to `n_features`)
- `d_model`: internal transformer dimension, auto-set to `2 * n_features`
- `d_feedforward`: hidden size of the feed-forward networks inside transformer layers (default 16)
- `dropout`: dropout rate applied throughout (default 0.1)
- `use_layer_norm`: toggles between reference-faithful layers (no LayerNorm) and paper-faithful layers (with LayerNorm)
- `dtype`: `"float32"` or `"float64"`, controlling model and data precision

The `d_model = 2 * n_features` sizing comes from the fact that the encoder concatenates two `n_features`-wide tensors (the input window and the focus score) along the feature dimension before feeding them into the transformer.

### Input Representation

Raw time series data is preprocessed before entering the model. In `scripts/preprocess_smd.py`, each feature is min-max normalized to $[0, 1)$ using training set statistics:

$$x_t \leftarrow \frac{x_t - \min(\mathbf{T})}{\max(\mathbf{T}) - \min(\mathbf{T}) + \epsilon'}$$

where $\min(\mathbf{T})$ and $\max(\mathbf{T})$ are per-feature vectors computed from the training series only. The same min/max values are applied to test data, meaning test values can fall outside $[0, 1)$ if the test distribution differs from training. These normalization parameters are saved as `{machine_id}_norm_params.npy` for use at inference time.

The normalized series is then converted into sliding windows via `convert_to_windows()` in `app/tranad_utils.py`. For each timestamp $t$, the window $W_t$ contains the $K$ most recent observations $\{x_{t-K+1}, \dots, x_t\}$. For timestamps $t < K$ where the full history is not available, the function uses replication padding by prepending copies of $x_0$. The implementation is vectorized using `torch.Tensor.unfold` rather than a Python loop.

The output shape is `(N, K, F)` where $N$ is the number of timestamps, $K$ is the window size, and $F$ is the number of features. During the forward pass, windows are permuted to `(K, N, F)` (sequence-first format expected by PyTorch's transformer modules), and the last element of each window is extracted as `elem` with shape `(1, N, F)` to serve as the reconstruction target.

### Positional Encoding

The `PositionalEncoding` module in `tranad_model.py` adds sinusoidal position information to the input embeddings. There is a quirk in the implementation that differs from the standard formulation in Vaswani et al. (2017). The standard approach interleaves sine and cosine across alternating dimensions:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

The reference implementation instead sums sine and cosine into the same dimensions:

```python
pe += torch.sin(position * div_term)
pe += torch.cos(position * div_term)
```

This means each dimension receives $\sin + \cos$ rather than one or the other. The `div_term` computation also uses `torch.arange(0, d_model)` rather than `torch.arange(0, d_model, 2)`, so the frequency spacing differs from the standard formulation. Our implementation preserves this reference behavior. The practical impact is minimal since the model learns to interpret whatever positional signal it receives, but it means the encoding is not a bijective mapping of positions in the way the original transformer paper intended.

### Transformer Layers

The implementation provides two variants of each layer type, selectable via `config.use_layer_norm`.

**Default variant (reference-faithful, `use_layer_norm=False`):** `TransformerEncoderLayer` and `TransformerDecoderLayer` use residual connections with dropout but no LayerNorm. The activation function is `LeakyReLU` rather than the `ReLU` used in standard transformers. This matches the reference code at `tranad/dlutils.py`.

The encoder layer performs:

```
src = src + dropout(self_attn(src, src, src))
src = src + dropout(ffn(src))
```

The decoder layer adds a cross-attention step between self-attention and the feed-forward network:

```
tgt = tgt + dropout(self_attn(tgt, tgt, tgt))
tgt = tgt + dropout(cross_attn(tgt, memory, memory))
tgt = tgt + dropout(ffn(tgt))
```

**Paper-faithful variant (`use_layer_norm=True`):** `TransformerEncoderLayerLN` and `TransformerDecoderLayerLN` add `nn.LayerNorm(d_model)` after each residual connection, matching Equations 4 and 5 in the paper:

```
src = LayerNorm(src + dropout(self_attn(src, src, src)))
src = LayerNorm(src + dropout(ffn(src)))
```

The paper explicitly describes LayerNorm in its equations, but the authors' own code omits it. Sweep results found that `use_layer_norm=False` (the reference behavior) performed comparably, so it remains the default.

**Notable omission: causal masking.** The paper (Eq. 5) mentions masking future positions in the window encoder to prevent the decoder from seeing future timestamps during parallel training. The reference code does not implement this mask, and neither does this implementation. The paper states this is needed because "all data $W$ and $C$ is given at once to allow parallel training," but in practice the window size is small (10 timesteps) and each window's reconstruction target is only its last element, so the masking has limited practical effect.

### Two-Phase Self-Conditioning (Forward Pass)

The core architectural novelty of TranAD is the two-phase forward pass implemented in `TranADNet.forward()`. The model runs the encoder-decoder pipeline twice per input, with the second pass conditioned on the errors from the first.

**Phase 1** starts with a zero focus score. The input window $W$ (shape `(K, N, F)`) is concatenated with a zero tensor of the same shape along the feature dimension, producing a tensor of shape `(K, N, 2F)`. This is scaled by $\sqrt{F}$, passed through positional encoding, and fed to the transformer encoder to produce a memory representation. The reconstruction target `elem` (the last timestep, shape `(1, N, F)`) is doubled to `(1, N, 2F)` to match the encoder output dimension, then passed through Decoder 1 along with the encoder memory. A final linear projection (`nn.Linear(2F, F)`) followed by sigmoid produces the Phase 1 reconstruction $O_1$ in the range $[0, 1]$.

**Phase 2** computes the focus score as the squared element-wise error from Phase 1: $F = (O_1 - W)^2$. This replaces the zero tensor from Phase 1. The concatenation, encoding, and decoding steps are repeated, but this time using Decoder 2 instead of Decoder 1. The focus score acts as an attention prior, giving the encoder higher activation on sub-sequences where Phase 1 performed poorly. The output is the Phase 2 reconstruction $\hat{O}_2$.

In code:

```python
def forward(self, src, tgt):
    # Phase 1: zero focus score
    c = torch.zeros_like(src)
    x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))

    # Phase 2: focus score from Phase 1 error
    c = (x1 - src) ** 2
    x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

    return x1, x2
```

The encoder is shared between both phases; only the decoders are separate. This means the two decoders receive the same architectural capacity but learn different roles during adversarial training (when enabled).

**Distinction from the paper:** The paper's Figure 1 and Algorithm 1 suggest that the focus score from Phase 1 is $\|O_1 - W\|_2$ (the L2 norm of the reconstruction error). The implementation uses $(O_1 - W)^2$ (the element-wise squared error without the norm reduction), which preserves the per-dimension and per-timestep structure of the error signal. This is a richer signal than a scalar norm because it tells the encoder which specific dimensions and timesteps had high error, not just the aggregate magnitude. The reference code uses this same element-wise formulation.

### Float64 Support

The reference code operates in float64 throughout (calling `.double()` on the model). This implementation defaults to float32 for faster training and broader device compatibility (MPS does not fully support float64), with float64 available via `config.dtype = "float64"`. When float64 is selected, `TranADNet.__init__` calls `self.double()` to convert all parameters. The scoring pipeline in `TranADScorer.score_batch` auto-detects the model's dtype via `next(model.parameters()).dtype` and casts input data accordingly.

---

## Scoring System

### Overview

The scoring system converts raw model outputs (reconstruction errors) into binary anomaly labels. It operates in three stages: score computation, threshold calibration, and label assignment. All scoring logic lives in `app/tranad_scorer.py`, with the SPOT algorithm in `app/spot.py`.

### Score Computation

`TranADScorer.score_batch()` runs the trained model over a dataset and returns per-dimension MSE scores. For each window, the model produces its two-phase outputs $O_1$ and $\hat{O}_2$, and the score is computed against the reconstruction target (the last element of the window).

Two scoring modes are available, controlled by `scoring_mode`:

**`phase2_only` (default, reference code behavior):** The anomaly score is the MSE between the Phase 2 output and the target:

$$s_t = \text{MSE}(\hat{O}_2, W_t)$$

This is what the reference code uses (`z[1]` in `tranad/main.py`).

**`averaged` (paper Eq. 13):** The anomaly score is the average of both phases:

$$s_t = \frac{1}{2}\|O_1 - W_t\|_2 + \frac{1}{2}\|\hat{O}_2 - W_t\|_2$$

Sweep results found that `averaged` mode generally improves precision without sacrificing recall, particularly on machine-3-2 where it boosted F1 from 0.980 to 0.987. The improvement comes from incorporating the Phase 1 signal, which captures different error characteristics than Phase 2 alone.

The output of `score_batch` is a 2D array of shape `(N, F)` containing per-dimension scores. For threshold calibration and detection, these are aggregated to 1D by taking the mean across features: `score_1d = np.mean(test_scores, axis=1)`. For diagnosis (root cause attribution), the per-dimension scores are used directly.

### Threshold Calibration

Threshold calibration is a post-processing step that determines the boundary between normal and anomalous scores. It runs once after training, requires no anomaly labels (for POT and percentile methods), and produces a single scalar threshold that is saved for use at inference time.

Three methods are implemented in `calibrate_threshold()`:

#### POT (Peaks-Over-Threshold)

This is the primary method, based on Extreme Value Theory. The implementation in `app/spot.py` is adapted from the reference code, which itself is based on the SPOT algorithm by Siffer et al. (KDD 2017).

The calibration process works as follows. Training scores (1D, mean-aggregated) are used as the initial calibration data. SPOT sorts these scores and picks an initial threshold at the specified `level` percentile. For example, at `level=0.99`, the initial threshold is the 99th percentile of training scores. All training scores exceeding this initial threshold are collected as "excesses" (the amount by which they exceed the threshold). These excesses are fit to a Generalized Pareto Distribution (GPD) using the Grimshaw maximum likelihood estimator. The fitted GPD parameters $\gamma$ (shape) and $\sigma$ (scale) define the tail behavior of the score distribution. From these, a quantile is computed at the specified risk level $q$:

$$\text{threshold} = t_0 + \frac{\sigma}{\gamma}\left(\left(\frac{nq}{N_t}\right)^{-\gamma} - 1\right)$$

where $t_0$ is the initial threshold, $n$ is the total number of calibration points, and $N_t$ is the number of excesses. The threshold is then multiplied by a `scale` factor (a per-machine tuning parameter).

The `level` parameter has a large impact on the resulting threshold. A low level like 0.95 includes the top 5% of training scores in the tail model, producing a relatively low threshold. A high level like 0.99995 restricts the tail to only the most extreme 0.005%, producing a much higher threshold. The appropriate level depends on the machine's score distribution. Machine-2-1, for example, has a long right tail even on normal data, so a low level (0.95 from the reference) incorrectly classified too much normal data as the tail, producing a threshold so low that 63% of test data was flagged. Raising the level to 0.999 resolved this.

**Robustness fallback.** A critical addition over the reference code is the anomaly-rate validation check. After computing the POT threshold, the scorer checks what fraction of test data would be flagged:

```python
anomaly_rate = float(np.mean(test_scores_1d > pot_th))
if anomaly_rate > 0.20:
    pot_th = float(np.percentile(test_scores_1d, 99.9))
```

If more than 20% of test data would be classified as anomalous, the threshold is considered unreliable (real anomaly rates in SMD are under 12%), and the scorer falls back to the 99.9th percentile of test scores. This was essential for machine-3-7, which has a 144x ratio between mean test scores and mean training scores, causing standard POT to set a threshold far too low.

**Retry loop.** SPOT's `initialize()` method can fail numerically if the `level` is too extreme for the data (e.g., the selected initial threshold has zero excesses). The implementation includes a retry loop that reduces `level` by a factor of 0.999 on each failure, up to 1000 attempts:

```python
lms = pot_params.level
for _ in range(1000):
    try:
        s = SPOT(pot_params.q)
        s.fit(train_scores_1d, test_scores_1d)
        s.initialize(level=lms, min_extrema=False, verbose=False)
    except Exception:
        lms = lms * 0.999
    else:
        break
```

This comes directly from the reference code at `tranad/pot.py`.

#### Percentile

A simpler alternative that sets the threshold as a fixed percentile of the training score distribution:

$$\text{threshold} = P_k(\text{train\_scores})$$

where $k$ defaults to 99. This requires no test data and no EVT assumptions, making it more robust but less adaptive to the tail shape. It serves as a useful baseline and as the fallback target in the POT anomaly-rate check.

#### F1-Max

A brute-force search over candidate thresholds that maximizes the F1 score on labeled test data. This requires ground truth labels, so it cannot be used in a truly unsupervised production setting, but it is useful for establishing an upper bound on detection performance and for hyperparameter tuning during development.

The search evaluates `step_num` (default 100) evenly-spaced thresholds between the minimum and maximum test scores, applying the point-adjustment protocol at each, and returns the threshold with the highest F1.

### Per-Machine Threshold Tuning

Each machine is calibrated independently with its own POT parameters, stored in `scorer_state.json` via the `TranADRegistry`. The current per-machine settings are:

| Machine | `level` | `scale` | Notes |
|---------|---------|---------|-------|
| machine-1-1 | 0.99995 | 1.06 | Reference values, works well |
| machine-2-1 | 0.999 | 0.9 | Tuned from reference's 0.95 |
| machine-3-2 | 0.99 | 1.0 | Reference values, works well |
| machine-3-7 | 0.99995 | 1.06 | Anomaly-rate fallback typically activates |

This per-machine tuning is standard practice. The paper's reference code also uses per-machine POT parameters (in `src/constants.py`), though this is not emphasized in the paper itself. In production, these parameters would be determined during an onboarding calibration step for each new machine.

### Point-Adjustment Evaluation Protocol

The `_adjust_predicts()` method implements the standard evaluation protocol used across TSAD benchmarks (OmniAnomaly, USAD, TranAD, and others). The core idea is that for contiguous anomaly segments, detecting any single point within the segment counts as detecting the entire segment.

The algorithm works as follows. Binary predictions are initially generated by comparing scores against the threshold. Then, for each contiguous anomaly segment in the ground truth, if at least one point within that segment is correctly predicted as anomalous, all points in the segment are retroactively marked as correctly detected. This means a single true positive within a segment converts all false negatives in that segment into true positives.

This protocol reflects the practical reality that in time series anomaly detection, identifying that an anomaly occurred matters more than pinpointing the exact start and end timestamps. It tends to boost recall substantially (most machines achieve recall near 1.0 after point-adjustment), meaning F1 scores are primarily driven by precision (the false positive rate).

### Root Cause Diagnosis

For multivariate data, TranAD can identify which specific dimensions are anomalous at each timestamp by examining the per-dimension scores (before aggregation to 1D). The `diagnose()` method computes two ranking-quality metrics:

**HitRate@P%** measures what fraction of the true anomalous dimensions appear in the model's top-ranked dimensions. For each anomalous timestamp, if $k$ dimensions are truly anomalous, HitRate@100% checks whether all $k$ appear in the model's top $k$ dimensions (ranked by score). HitRate@150% gives the model 50% more candidates ($\lceil 1.5k \rceil$ dimensions) to find all $k$ true anomalies.

**NDCG@P%** (Normalized Discounted Cumulative Gain) measures the ranking quality of the model's dimension-level scores against the ground truth, penalizing cases where true anomalous dimensions are ranked lower. This uses `sklearn.metrics.ndcg_score` under the hood.

Both metrics operate at the per-timestamp level and are averaged across all anomalous timestamps.

---

## Training Pipeline

### Overview

Training is orchestrated by three scripts with increasing scope: `train_smd.py` trains a single machine, `evaluate_smd.py` scores and calibrates a trained model, and `train_all_machines.py` runs both across all four reference SMD machines and produces a comparative summary.

Each machine is trained as a completely independent model. There is no parameter sharing, transfer learning, or joint training across machines. The only shared element is the hyperparameter configuration (learning rate, loss weighting, etc.), which is passed uniformly through the CLI. Threshold calibration, however, uses per-machine POT parameters.

### Data Preparation

`preprocess_smd.py` loads raw SMD text files and produces normalized `.npy` arrays. For each machine:

1. Training and test data are loaded from separate text files (comma-separated, one row per timestamp, one column per feature).
2. Per-feature min and max values are computed from the training data only.
3. Both training and test data are normalized to $[0, 1)$ using these values (Eq. 1 in the paper).
4. Interpretation labels (which specify anomalous dimension ranges) are parsed into a binary matrix of shape `(N_test, F)`.
5. All arrays are saved to `data/smd/processed/`.

The interpretation label format is `start-end:dim1,dim2,...` with 1-indexed positions. The parser converts to 0-indexed with half-open intervals (`start-1:end-1` for rows, `int(i)-1` for dimensions).

### Windowing

At training time, `train_smd.py` converts the normalized data to sliding windows using `convert_to_windows()`. The resulting tensor has shape `(N, K, F)` where each row is a window of $K=10$ consecutive timesteps. These windows are wrapped in a `TensorDataset` and served via a `DataLoader` with configurable batch size (default 128).

When early stopping is enabled (`early_stopping_patience > 0`), the windows are split chronologically into training and validation sets using `val_split` (default 0.2). The split is not shuffled since temporal ordering matters for time series.

### Loss Computation

The training loop in `train_epoch()` implements the paper's evolving loss function with two configurable dimensions: the loss weighting schedule and the adversarial training mode.

#### Evolving Loss Weight

The weight $w$ controls the balance between Phase 1 (reconstruction) and Phase 2 (self-conditioned) loss terms. It starts high (prioritizing basic reconstruction when the model is untrained) and decreases over time (shifting emphasis to the adversarial/self-conditioned signal as reconstructions improve).

Two schedules are implemented in `compute_loss_weight()`:

**`epoch_inverse` (reference code):** $w = 1/n$ where $n$ is the epoch number (1-indexed). This decays rapidly: $w = 1.0, 0.5, 0.33, 0.25, \dots$ At epoch 20, $w = 0.05$, meaning 95% of the loss comes from the Phase 2 term.

**`exponential_decay` (paper Eq. 10):** $w = \epsilon^{-n}$ where $\epsilon$ defaults to 1.01. This decays much more slowly: at epoch 20, $w = 0.82$, meaning 82% of the loss still comes from Phase 1. This maintains a stronger reconstruction signal throughout training.

The difference is significant. `epoch_inverse` rapidly shifts all learning pressure to Phase 2, which can destabilize training if Phase 1 reconstructions haven't yet converged (the focus scores fed to Phase 2 are unreliable noise). `exponential_decay` keeps the Phase 1 signal dominant for much longer, which empirically produced substantially better results. Switching from `epoch_inverse` to `exponential_decay` alone improved machine-1-1 F1 from 0.795 to 0.921.

The $n$ counter increments per epoch, not per batch. Per-batch incrementing would cause $\epsilon^{-n}$ to reach near-zero within the first epoch for any reasonably-sized dataset, effectively disabling the Phase 1 loss entirely.

#### Non-Adversarial Mode (Default)

When `adversarial_loss=False`, both decoder outputs are combined into a single loss:

$$L = w \cdot \text{MSE}(O_1, W) + (1 - w) \cdot \text{MSE}(\hat{O}_2, W)$$

A single backward pass computes gradients for all parameters (encoder + both decoders), and one optimizer step is taken.

This is how the reference code operates. It does not implement the paper's minimax objective (Eq. 8). Both decoders are pushed to minimize reconstruction error, with the only difference being that Decoder 2 receives the focus-score-conditioned input.

#### Adversarial Mode (Paper Eq. 8-9)

When `adversarial_loss=True`, the two decoders receive separate losses with opposite signs on the Phase 2 term:

$$L_1 = w \cdot \text{MSE}(O_1, W) + (1 - w) \cdot \text{MSE}(\hat{O}_2, W)$$
$$L_2 = w \cdot \text{MSE}(O_1, W) - (1 - w) \cdot \text{MSE}(\hat{O}_2, W)$$

Decoder 1 minimizes $\text{MSE}(\hat{O}_2, W)$ (trying to produce a perfect reconstruction so the focus score is zero). Decoder 2 maximizes it (trying to amplify the reconstruction error). The shared encoder sees partially-canceling gradients from the two backward passes, creating a GAN-like equilibrium.

The implementation performs two sequential backward passes (`l1.backward(retain_graph=True)` then `l2.backward()`) before a single optimizer step. The `retain_graph=True` is necessary because both losses share the computation graph through the encoder.

In practice, adversarial training proved unstable for downstream thresholding. While it can produce very high F1 values in isolated evaluations (e.g., 0.999 on machine-1-1 in sweep trial 29), it creates extreme score distributions where test scores reach orders of magnitude above training scores (e.g., test max of ~100M vs training scores near 0). These extreme ranges break POT calibration. The adversarial mode is therefore disabled by default.

### Gradient Clipping

After the backward pass(es), gradient norms are clipped when `gradient_clip_norm > 0` (default 1.0):

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
```

This is particularly important when adversarial training is enabled, where the two backward passes can produce large opposing gradients. Even in non-adversarial mode, clipping provides a safety net against occasional gradient spikes.

### Early Stopping

The `EarlyStopping` class monitors validation loss and halts training when it stops improving. On each call to `step()`, if the validation loss is lower than the previous best, the model's state dict is cloned and the patience counter resets. If the loss fails to improve for `patience` consecutive epochs, training stops and the best weights are restored.

When early stopping is disabled (`patience=0`), training runs for the fixed `epochs` count without a validation split. When enabled, training runs for up to `max_epochs` (default 50) with the training data split 80/20 for train/validation.

Sweep results showed that early stopping primarily prevented wasted computation on divergent configurations (adversarial trials at high learning rates typically stopped at epoch 4). Well-behaved configurations at low learning rates (0.0001) rarely triggered early stopping within 30 epochs.

### Optimizer and Scheduler

The optimizer is AdamW with configurable learning rate (default $10^{-4}$) and weight decay ($10^{-5}$). A StepLR scheduler reduces the learning rate by a factor of `scheduler_gamma` (0.9) every `scheduler_step` (5) epochs. This provides a mild annealing effect over training.

The paper reports using an initial learning rate of 0.01 with a step scheduler of step size 0.5, and a meta-learning rate of 0.02. The implementation defaults to a much lower learning rate (0.0001) because sweep results showed that higher rates (0.001, 0.01) frequently caused training instability, particularly on machines other than machine-1-1. The paper's higher learning rate may have been stabilized by MAML meta-learning (which is not implemented) or by other aspects of their training setup.

### Checkpoint Saving

After training completes (either via epoch limit or early stopping), the model is saved as a PyTorch checkpoint to `models/tranad/{machine_id}/model.ckpt`. The checkpoint contains the model state dict, optimizer state, scheduler state, the full `TranADConfig`, the final epoch number, and the final training loss. The `TranADRegistry` in `app/tranad_registry.py` manages loading these checkpoints with automatic caching.

### Evaluation Pipeline

`evaluate_smd.py` loads a trained checkpoint and runs the full scoring pipeline:

1. The model is loaded via `TranADRegistry.get_model()`, which deserializes the config and weights and sets the model to eval mode.
2. Both training and test data are scored using `TranADScorer.score_batch()`, producing per-dimension MSE arrays.
3. A threshold is calibrated using the specified method (POT by default) via `calibrate_threshold()`. For POT, training scores serve as the calibration data and test scores as the stream. The anomaly-rate fallback provides robustness.
4. Detection metrics (F1, precision, recall, AUC) are computed via `evaluate()`, which applies the point-adjustment protocol.
5. Diagnosis metrics (HitRate, NDCG) are computed via `diagnose()` using the per-dimension scores.
6. Results are saved to `models/tranad/{machine_id}/eval_results.json` and the scorer state (threshold + method) to `scorer_state.json`.

### Multi-Machine Orchestration

`train_all_machines.py` runs the training and evaluation pipeline across all four reference SMD machines (machine-1-1, machine-2-1, machine-3-2, machine-3-7) as separate subprocesses. This design isolates each machine's training (preventing GPU memory leaks between runs) while reusing the existing single-machine scripts.

The script uses per-machine POT parameters from a hardcoded `POT_PARAMS` dictionary. After all machines complete, it prints a summary table comparing per-machine and average results against the paper's Table 2 values, and saves a summary JSON.

### Hyperparameter Sweep

`sweep_smd.py` performs a grid search over training and scoring parameters. Each trial builds a `TranADConfig` from the parameter combination, trains with early stopping, scores, calibrates, evaluates, and logs results to a CSV. The sweep supports two grid sizes (quick: 48 combinations, full: 480) and can resume from a partially-completed CSV.

Key sweep findings that informed the default configuration:

- `exponential_decay` loss weighting consistently outperformed `epoch_inverse` across machines.
- Adversarial training produced unstable results that did not survive standalone POT evaluation.
- `averaged` scoring mode improved precision on most machines.
- Low learning rate (0.0001) was the only consistently safe choice across machines.
- `d_feedforward` (16 vs 64) and `use_layer_norm` had modest impact.

### Deferred Components

Several components described in the paper are not yet implemented:

**MAML meta-learning** (paper Algorithm 1, line 11): The paper's ablation study shows ~1% F1 improvement on full data but ~12% on limited data (F1* metric). MAML requires computing second-order gradients ($\nabla_\theta L(f(\theta'))$ where $\theta'$ already depends on $\nabla_\theta$), adding significant complexity. It is deferred as low-impact for the current evaluation setting where full training data is available.

**Per-dimension thresholding** (paper Algorithm 2, lines 5-6): The paper's inference algorithm applies POT per dimension ($y_i = \mathbf{1}(s_i \geq \text{POT}(s_i))$) and then takes the logical OR across dimensions ($y = \bigvee_i y_i$). The current implementation and the reference code both aggregate to 1D first, then threshold once. Per-dimension thresholding could improve detection on machine-3-7, where two anomaly segments are undetectable after mean aggregation because only a few dimensions are anomalous while the rest mask the signal.

**Causal masking in the window encoder**: Described in Eq. 5 of the paper but omitted in the reference code. Expected to have minimal impact given the small window size.