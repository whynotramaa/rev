# Research on a choice-order-invariant decision model

Research date: 13 October 2026. Scope: a first-pass architecture and experiment plan using Qwen3-0.6B and Kaggle free-tier training. This report does not contain a trained model or measured GPU performance.

## 1. Recommended direction

Build a model that reads a state, instructions, and candidate descriptions, then produces one score per candidate in a single forward pass. Use the same scoring function for every candidate. Convert those scores into typed results in ordinary application code.

The strongest starting design for your constraints is:

- The full Qwen3-0.6B backbone, without its language-generation output head.
- A causal shared question/state prefix and isolated causal option branches, matching the attention graph in your third image.
- Position IDs that restart at the same base for each option, while preserving token order inside each option.
- A short final answer-summary span whose positions do not depend on option order.
- One small, position-free attention block over the option embeddings and summary, followed by a shared bilinear scorer.
- LoRA on attention projections in the upper 12 backbone layers, with the decision head trained normally.
- Cross-entropy, including genuine soft targets where available, followed by held-out temperature calibration.
- A first serious run on 100,000 distinct decisions, followed by expansion only when a learning curve justifies it.

The bilinear-only head is the required simpler baseline. Keep the small attention block only if it improves independent evaluation enough to justify its cost. The summary token already permits some comparison between options.

Start with the dataset you supplied. Generate additional data after you identify failures, especially rule exceptions, changed option definitions, missing evidence, and difficult distractors. Rephrasing the same questions many times should not be the first investment.

There is no defensible universal "optimal dataset size" or promise that a 0.6B model will beat frontier systems across all decisions. The achievable target is a competitive small model for bounded, grounded decisions, with measured accuracy, useful uncertainty, and an explicit order-symmetry guarantee. A focused support-routing demo is a stronger first deliverable than a universal decision engine.

## 2. What the problem actually is

### 2.1 The model's job

Let `s` be the state, `q` the instructions, and `C = {c₁, …, cₖ}` the candidate descriptions. The model estimates:

$$
p_i = P(c_i \text{ is the intended decision}\mid s,q,C).
$$

These probabilities are conditional on the supplied candidates and the task definition. They are not universal truth probabilities attached to the strings.

The intended interface has three tasks:

| Task | Model output | Application output |
| --- | --- | --- |
| `choice` | Distribution over the supplied candidates | A supplied candidate key and its distribution |
| `noul` | Distribution over false and true | The probability of true |
| `score` | Distribution over described ordinal levels | The expected level and its distribution |

The official Jev documentation defines this state-and-questions interface and evaluates questions independently. That public contract is useful to reproduce. It does not establish that your diagrams reproduce Jev's private internal network. [TypeSafe introduction](https://docs.typesafe.ai/introduction), [Choice](https://docs.typesafe.ai/primitives/choice), [Noul](https://docs.typesafe.ai/primitives/noul), [Score](https://docs.typesafe.ai/primitives/score).

A non-generative classifier prevents invented output keys when the application maps scores back to validated input candidates. It can still select the wrong candidate, misunderstand the state, or be overconfident. Type correctness and decision correctness need separate tests.

### 2.2 Equivariance is the precise requirement

For a permutation matrix `Π` that reorders candidates, require:

$$
f(s,q,\Pi C)=\Pi f(s,q,C).
$$

This is permutation equivariance of the score or probability vector. The probabilities follow their candidate identities. The selected semantic candidate is invariant when there is a unique winner.

For example, probabilities associated with Delhi, Mumbai, and Kolkata must remain associated with those cities after a shuffle. Expecting the raw output array to remain identical would be the wrong test.

There are several different robustness requirements:

| Change | Desired behavior | Guaranteed by the proposed symmetry? |
| --- | --- | --- |
| Reorder complete choice branches | Per-key logits and probabilities agree within numerical tolerance | Yes, in exact arithmetic |
| Rename an opaque output ID | Same distribution over descriptions | Yes, if IDs never enter the network |
| Paraphrase a description | Similar answer when meaning is preserved | No, requires data and evaluation |
| Add a plausible distractor | A sensible revised distribution | No, this changes the task |
| Duplicate a candidate | A defined policy for the duplicate | No, multiplicity changes the task |
| Reorder words within a candidate | Usually a different input | No, token order must remain meaningful |
| Reorder ordinal levels and change their values | Usually a different scoring rubric | No, level order has semantic meaning |

Set-based neural networks provide the general symmetry principles. Set-LLM specifically studies attention masks and positions for text containing unordered sets. Its published mask uses a different prefix treatment from the causal branch design recommended here. Its result supports the direction, but its benchmark results do not predict Qwen3-0.6B performance. [Deep Sets](https://arxiv.org/abs/1703.06114), [Set Transformer](https://arxiv.org/abs/1810.00825), [Set-LLM](https://arxiv.org/abs/2505.15433).

### 2.3 The three independent failure points in your diagrams

Your first image correctly identifies three places where shuffling can affect the answer:

1. An option's token positions change when an earlier option has a different length.
2. With ordinary causal attention, an option can read earlier options but cannot read later ones.
3. The head can attach meaning to a physical option slot, even after the backbone is repaired.

Resetting option positions fixes the first problem. Your third image fixes the second. A shared, position-free decision head fixes the third. All three fixes are necessary for this particular construction.

## 3. What existing work tells us

### 3.1 Jev is an interface reference, not a disclosed training recipe

TypeSafe describes Reinforcement Learning for Calibrated Decisions, or RLCD, and reports speed and calibration benefits. The official material inspected here does not give enough low-level architecture and optimization detail to reproduce that training method. Treat those as vendor claims and interface guidance, not an implementation specification. [TypeSafe launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

You do not need to reproduce RLCD to create a useful decision model. Supervised learning with probability-sensitive losses is the first experiment. A reward for picking the right answer alone would encourage confident argmax behavior without teaching honest probabilities.

### 3.2 BEV is unusually close to your design

The open BEV code provides the relevant implementation reference. Its encoder restarts positions, isolates options in the backbone, and appends a summary span. Its head uses option self-attention and shared projected dot products. The model card reports a 20-layer Qwen derivative, LoRA, and 74.7% accuracy on 5,000 held-out mixture questions. Those are author-reported results, not measurements reproduced for this report. [BEV encoder source](https://github.com/avbiswas/bev-decider/blob/aa5d861f256b57f32794715de2559cd5a7b68a30/src/bev_decider/encode.py), [BEV head source](https://github.com/avbiswas/bev-decider/blob/aa5d861f256b57f32794715de2559cd5a7b68a30/src/bev_decider/model.py), [BEV model card](https://huggingface.co/avbiswas/bev-decider-0.4B).

Use that code to understand the construction and establish a baseline. The inspected repository contains an inference implementation, not a complete reproduction of the model card's training pipeline. Its Apache-2.0 code and its CC-BY-NC-4.0 released weights have different permissions. Starting from the original Qwen weights gives you a cleaner experimental starting point; your chosen training data still determines release obligations. [BEV repository](https://github.com/avbiswas/bev-decider).

The report's proposed 28-layer backbone, smaller first head, description-only rendering, length policy, and training schedule are design recommendations. They are not settings proven superior to BEV.

### 3.3 Label text can defeat definition-following

A recent preprint finds option-label bias in several open typed-decision implementations. In its experiments, models sometimes follow short labels even when definitions specify different rules. Rendering only the definition removes a source of that bias in the systems studied. The finding motivates a targeted test here; it does not prove that every model behaves identically. [Labels Override Definitions in Jev-Style Typed Decision Models](https://arxiv.org/abs/2610.02586).

For this project, keep an opaque candidate ID in application code and put its description into the model. If the user provides no description, require a meaningful candidate name and use that as the description. Do not pretend that an opaque key such as `c17` has enough semantic information.

For a routing question, `refund_queue` is an output ID. The model should read "Requests for money returned after a duplicate charge." An ID rename should then have no effect at all.

Other useful comparison systems include GLiClass, which offers several joint and separate encoder constructions. Use such systems as accuracy-and-latency baselines. Do not assume that their default position handling guarantees choice-order symmetry. [GLiClass repository](https://github.com/Knowledgator/GLiClass).

## 4. The recommended architecture

### 4.1 Backbone and serialization

Assume your "Qwen 0.6B" means `Qwen/Qwen3-0.6B`. Confirm that identifier before implementation; a different checkpoint needs a new configuration and tokenizer audit. The inspected config has 28 layers, hidden size 1,024, 16 query heads, 8 key/value heads, head dimension 128, and RoPE. The checkpoint is Apache-2.0. [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B), [pinned config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json).

Use `Qwen3Model` or the corresponding backbone loaded from the pretrained checkpoint. A decision loss does not need vocabulary logits. Qwen ties its language-model weights to input embeddings, so removing the output operation does not remove those necessary input embeddings.

The physical sequence is:

```text
Q = task hint + instructions + state
C1 = identical opening delimiter + description 1 + identical closing delimiter
C2 = identical opening delimiter + description 2 + identical closing delimiter
...
CK = identical opening delimiter + description K + identical closing delimiter
A = a short fixed summary span
```

Put instructions before the state so state tokens can encode the question under causal attention. This makes representations question-conditioned. It also means different questions about the same state cannot directly reuse the same complete prefix cache. That tradeoff is explicit.

Start with existing tokenizer tokens for delimiters. A string such as `<option>` may be several tokens; measure the actual span and do not assume it is one token. Record every branch's start, end, and readout offset during tokenization.

Keep candidate descriptions independent of display order. Do not prepend `A`, `B`, "first option", or a serial counter. Parse JSON states and serialize objects with stable key ordering. Preserve list order because lists can carry sequence semantics. This removes avoidable JSON-layout variation, but it does not make arbitrary text paraphrases equivalent.

### 4.2 Position IDs

Let `P` be the number of prefix tokens, `ℓᵢ` the full token length of branch `i`, and `a` the summary length. Include delimiters in `ℓᵢ`.

Assign positions as follows:

$$
\operatorname{pos}(Q_t)=t,\qquad 0\le t<P.
$$

$$
\operatorname{pos}(C_{i,t})=P+t,\qquad 0\le t<\ell_i.
$$

$$
\operatorname{pos}(A_t)=P+\max_i\ell_i+t,\qquad 0\le t<a.
$$

Every option starts at `P`, exactly as your second image proposes. Tokens inside an option retain different positions. Setting every token to the same position would erase useful within-option ordering.

The summary starts after the longest logical branch. The longest length is unchanged by permutation. Avoid using the last physical branch's end position, which varies when branch lengths differ.

Physical storage indices and RoPE position IDs are different concepts. A packed sequence can have repeated position IDs while its tensor indices remain unique. If you later introduce KV caches, do not use repeated RoPE IDs as cache-write addresses.

### 4.3 Attention mask

Define `M[u,v] = 1` when query token `u` may attend to key token `v`.

| Query segment | Allowed key segments | Rule inside its own segment |
| --- | --- | --- |
| `Q` | `Q` | Causal |
| `Cᵢ` | All of `Q`, and `Cᵢ` | Causal inside `Cᵢ` |
| `A` | All of `Q`, all real choices, and `A` | Causal inside `A` |

Every other real-token edge is blocked. In particular, `Cᵢ` cannot attend to `Cⱼ` for `i ≠ j`, and `Q` cannot attend to a choice or summary token.

Your third image is the right graph for this design. Apply it in every backbone layer. A conventional attention layer inserted midway would break the isolation.

Use an additive mask with shape `[batch, 1, L, L]`, containing zero for allowed entries and negative infinity for blocked entries. The head needs a separate candidate-validity mask. A standard `[batch, L]` padding mask cannot express branch isolation.

For padding, real queries must never attend to padded keys. Give each padded query a harmless self-edge, then exclude padded outputs from every readout and loss. This avoids softmax rows with no valid entries. A request with zero real options must be rejected before the network runs.

At the direct PyTorch SDPA boundary, boolean `True` means allowed. Other attention APIs use `True` to mean blocked. Use one internal convention and convert explicitly at the boundary. Pass the complete custom mask without allowing a backend to replace it with ordinary causality. [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

Transformers mask handling changes across versions and attention backends. Begin with eager attention as a correctness reference, then check SDPA against it. A 4D mask may change which fused kernel runs; a masked computation is not automatically a sparse computation. Current Transformers documentation also warns that custom backend registrations need matching mask handling. [Transformers attention backends](https://huggingface.co/docs/transformers/main/en/attention_interface).

The Qwen and SDPA source inspected at Transformers `v4.57.1` supports examining this explicit-mask path. That is a source-inspected candidate version, not a GPU-tested environment for this report. Pin an exact working set of PyTorch, Transformers, PEFT, tokenizer, and CUDA versions after the first smoke test. [Qwen source at v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py), [SDPA integration at v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/sdpa_attention.py).

### 4.4 Readouts and the decision head

Read each option at the last token of its closing delimiter. The entire delimiter span is identical across choices, and this token can see the full description. Compare this with reading the last content token in a small ablation; do not add a learned special token initially.

Let `hᵢ ∈ R^{1024}` be the option readout and `h_A` the final summary readout.

The minimal bilinear baseline is:

$$
u=W_A\operatorname{LN}(h_A),\qquad
v_i=W_C\operatorname{LN}(h_i),\qquad
z_i=\frac{u^T v_i}{\sqrt d}.
$$

Use `d = 256` initially. Both projection matrices are shared across all candidates. This is a low-rank bilinear score, with approximately `2 × 1024 × 256 = 524,288` projection weights, excluding small bias and normalization terms.

A full `1024 × 1024` bilinear matrix has about 1.05 million weights. It is also valid if shared across choices. The smaller factorization is the first experiment, not a claim that rank 256 is optimal.

For the proposed small comparison head:

1. Project option and summary readouts to width 256 with a shared input projection.
2. Form `[task embedding, option embeddings, summary embedding]`.
3. Apply one pre-normalized self-attention block, four heads, and feed-forward width 512.
4. Use no position encodings, option-slot embeddings, or causal mask in this head.
5. Mask invalid candidate slots as keys and exclude them from scoring and the loss.
6. Apply shared bilinear projections to the updated summary and options.

The task and summary tokens are distinguished by role, while option tokens all have the same role. Reordering only the option tokens therefore preserves the intended symmetry. One block of this size should keep the new trainable head around a million parameters, depending on projection choices. Count the actual parameters in implementation.

Comparison belongs in the position-free head, where every choice has symmetric access to the others. Independent option scoring is a worthwhile baseline, especially for large catalogs. Some tasks genuinely depend on the candidate set, such as choosing the most appropriate handler among overlapping definitions.

### 4.5 A variable number of choices

Do not use `Linear(hidden_size, MAX_CHOICES)` as the general decision classifier. Its output columns have learned slot identities and do not describe arbitrary new candidates.

Instead, score the supplied candidate embeddings. Pad candidate tensors to the largest `K` in the current batch and apply:

$$
\tilde z_i=\begin{cases}z_i&i\text{ is valid},\\-\infty&i\text{ is padding},\end{cases}
\qquad
p=\operatorname{softmax}(\tilde z).
$$

Your idea of masking excess columns is correct as a batching rule. It does not require a fixed collection of learned output columns.

Set a service resource limit of 32 candidates initially. The audited dataset has at most 17 choices, so 18 to 32 must be treated as an extrapolation range until additional data and tests support it. Report accuracy separately by candidate count. A first published demo can limit itself to 2 to 16 choices.

The initial API should require at least two candidates. A single-candidate softmax always returns 1 and says nothing about whether that candidate is actually appropriate.

### 4.6 Why the construction has the symmetry

Consider permuting complete option branches, with each branch's text and metadata moving together.

1. The prefix's input and allowed attention edges are unchanged, so its representations are unchanged.
2. Each option receives the same local positions and can attend to the same prefix and its own token sequence. Its representation therefore moves with its identity.
3. The summary's positions are unchanged. Its attention inputs are permuted key/value pairs with matching representations and positions. Attention sums are unchanged in exact arithmetic.
4. A position-free set-attention head is equivariant over the candidate tokens and invariant at the distinguished summary token.
5. Shared bilinear scoring and masked softmax preserve the candidate permutation.

Thus the output probabilities reorder with the candidates. Unequal option lengths do not change this argument.

This proof assumes that tokenization, preprocessing, every layer, and the head obey these rules. Dropout introduces sample-level randomness during training. Test symmetry with `model.eval()` and attention dropout zero. Floating-point reduction order can still produce small discrepancies, including in fp32. Do not describe fp32 as bitwise exact.

For an exact tie, select the smallest stable output ID in application code. This makes selection independent of list order for unchanged IDs. An ID-based tie rule cannot also be invariant to arbitrary ID renaming; report ties separately.

## 5. Typed tasks, probability losses, and uncertainty

### 5.1 One network can serve all three tasks

For `choice`, each description becomes a branch. The application returns the stable input ID associated with the winning score.

For `noul`, create false and true branches with explicit negative and positive descriptions where supplied. Return the probability associated with true. Avoid making a false-branch description by blindly prepending "not" to arbitrary text.

For `score`, attach each level's semantic rank and description to its branch. Compute:

$$
\hat r=\sum_{j=0}^{K-1} j\,p_j.
$$

The rank is semantic input for ordinal scoring, not a physical packed-slot position. If branches are reordered internally, their ranks must move with them. Changing the externally ordered rubric changes the question.

The expected level can be fractional. It is not a measured physical quantity, an interval-scale rating justified by the labels, or a probability. Keep the distribution available so a mean of 2 cannot hide a mixture of extreme levels.

### 5.2 The first training objective

For a hard label `y`, use cross-entropy over valid candidates:

$$
\mathcal L_{\mathrm{hard}}=-\log p_y.
$$

For a provided target distribution `t`, use:

$$
\mathcal L_{\mathrm{soft}}=-\sum_{i\in\mathrm{valid}} t_i\log p_i.
$$

Validate that each target is finite, nonnegative, aligned to candidate identities, and sums to one within a documented rounding tolerance. Renormalize only small rounding errors. Reject materially invalid targets. Do not multiply a padded zero target by a negative-infinite log probability; select valid entries before the reduction.

The `skills` data contains both mapping-shaped and list-shaped `label_probs`. Map dictionary keys to candidate identities. List targets must retain their original candidate or level alignment through any internal shuffle. Use the distribution instead of discarding it in favor of its argmax label.

Human annotator shares, a population outcome frequency, and a probability from an exactly specified random experiment can refer to different quantities. Preserve that meaning in the question and evaluate them separately. A teacher model's reported confidence is not an interchangeable ground-truth soft label.

Probability losses are the right starting objective, but finite data, a restricted model, and imperfect optimization do not guarantee calibration. [When Does Optimizing a Proper Loss Yield Calibration?](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e4165c96702bac5f4962b70f3cf2f136-Abstract-Conference.html).

Avoid class weighting, focal loss, and uniform label smoothing initially. They change the probability-learning problem. If you deliberately rebalance the training distribution, calibrate and evaluate on the intended deployment distribution.

### 5.3 An optional ordinal objective

Cross-entropy is enough for the first multi-task run. If ordinal errors remain large, compare it with an additional ranked probability score:

$$
\mathrm{RPS}(p,t)=\frac1{K-1}\sum_{r=0}^{K-2}
\left(\sum_{j=0}^{r}p_j-\sum_{j=0}^{r}t_j\right)^2.
$$

Try `CE + 0.1 × RPS` only on score questions, against the unchanged CE baseline. The coefficient is an experimental starting value. Do not optimize only the expected score's squared error, because many very different distributions share the same mean.

### 5.4 Calibration and review decisions

Reserve a calibration partition that never updates model weights. Fit a positive scalar temperature by minimizing held-out negative log-likelihood:

$$
p_i^{\mathrm{cal}}=\operatorname{softmax}(z_i/T).
$$

One shared temperature per task type is a reasonable first choice if each type has enough calibration examples. A single global temperature is the smaller fallback. Avoid slot-specific calibration parameters, because candidate slots have no meaning. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

Choose action thresholds on a separate policy-validation partition, then report performance on a sealed test set. A top probability of 0.95 becomes useful only after measuring how such predictions behave on relevant examples.

For a Jev-style compatibility layer, Choice confidence is a deterministic concentration statistic:

$$
\mathrm{confidence}_{\mathrm{choice}}
=\frac{p_{\max}-1/K}{1-1/K},\qquad K>1.
$$

Official Score confidence uses ordinal distances, and Noul has no separate confidence field. Implement the exact documented formulas only if matching that contract. A concentration statistic is not an independently trained probability of correctness. [TypeSafe confidence](https://docs.typesafe.ai/confidence).

For your initial product, retain calibrated probabilities, an application `review_required` flag, and the raw distribution. If a compatible `confidence` field is included, describe its formula and do not silently substitute a different statistic.

A model cannot express "none of the candidates fit" through softmax alone. Include an explicit unknown or review candidate when the task allows that outcome, and train examples where it is correct. A confident but incomplete candidate set is a real failure mode.

## 6. Dataset audit and implications

### 6.1 What was inspected

The audit downloaded all ten Parquet files at dataset revision `11ed9b5b879e7f9524e7c89c1037d498c54206f0`. It read every row, parsed every `questions_json`, checked hard-label membership and score bounds, and counted individual questions.

The five columns are `state`, `questions_json`, `domain`, `question_types`, and `question_count`. One row can contain multiple training decisions. The following counts are direct measurements of the pinned files, rather than estimates copied from a headline dataset size. [Pinned BEV dataset files](https://huggingface.co/datasets/avbiswas/bev-decision/tree/11ed9b5b879e7f9524e7c89c1037d498c54206f0).

| Config | Train rows | Train decisions | Test rows | Test decisions |
| --- | ---: | ---: | ---: | ---: |
| `default` | 125,614 | 250,262 | 24,386 | 46,320 |
| `hard_50k` | 42,000 | 75,046 | 8,000 | 14,479 |
| `numeric_temporal` | 16,477 | 32,848 | 3,329 | 6,518 |
| `skills` | 34,627 | 51,781 | 6,755 | 10,316 |
| `counterfactual_15k` | 12,614 | 12,614 | 2,386 | 2,386 |
| Total | 231,332 | 422,551 | 44,856 | 80,019 |

There are 276,188 rows and 502,570 decisions across both splits. The training decisions comprise 189,207 choice, 176,181 noul, and 57,163 score questions. These counts precede your own validation split, deduplication, and length filtering.

The audit found no malformed hard-label structures under those checks. A separate audit of all 8,131 soft targets found valid candidate keys or list lengths, finite values in the range zero to one, and probability sums within approximately `1.1e-6` of one. These checks do not establish that every label is semantically correct.

The `skills` config contains 6,851 soft-target training decisions and 1,280 soft-target test decisions. A count of questions is different from the dataset card's count of rows carrying soft targets.

### 6.2 Measured length distribution

A reservoir sample selected 5,000 training rows across the five files, in the table's order, with Python random seed 42. All 8,938 questions in those sampled rows were serialized for length measurement with the pinned Qwen tokenizer. The sample contained 32,728 option descriptions.

For the complete input, the measurement used a task hint, instructions before state, repeated `<option>` and `</option>` strings, descriptions without choice IDs, ranked score descriptions, and the summary "The answer is:". No text was truncated. These are sample statistics for that proposed template, not corpus-wide token totals.

| Quantity | Mean | Median | 90th percentile | 95th percentile | 99th percentile | Sample maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| State tokens | 274.13 | 97 | 877 | 1,238 | 2,166 | 4,321 |
| Instruction tokens | 17.88 | 13 | 36 | 43 | 63 | 101 |
| Option-description tokens | 6.82 | 3 | 16 | 24 | 30 | 431 |
| Complete packed-input tokens | 386.88 | 194 | 1,015 | 1,407 | 2,678 | 4,377 |

Among sampled complete inputs, 19.97% exceeded 512 tokens, 9.82% exceeded 1,024, and 2.39% exceeded 2,048. A 512-token limit would discard a substantial tail.

Use a 1,024-token total packed-input cap for the pilot and length buckets within that cap. Mark longer examples as excluded, preserve them for a separate long-input experiment, and report results for both short and long subsets. Move to 2,048 only after memory and throughput measurements justify it.

A 64-token per-option cap covers the typical options, but the measured maximum shows that it is not universal. Reject or explicitly exclude over-limit examples. Silently chopping a negation, exception, or distinguishing phrase can change the correct label.

For both training and the first API, reject oversized complete inputs rather than quietly returning a normal-looking answer to incomplete evidence. Later, an evidence-selection stage can support long states, with its own recall tests.

### 6.3 Leakage checks and their limits

The full audit hashed states after lowercasing and collapsing whitespace. It found 205,392 distinct training states and 40,749 distinct test states, with zero overlap between those sets.

That is useful, but it does not prove source independence. Related documents, paraphrases, subquestions, and generator templates can still cross a split. The dataset card says row-level provenance and some pair IDs are omitted, and describes several source-overlap caveats. Its test set is a held-out part of the mixture, not a fully independent benchmark. [BEV dataset card](https://huggingface.co/datasets/avbiswas/bev-decision).

Within the provided training split, keep all decisions for a normalized state in the same partition. Where you can recover related-source groups or paraphrase families, use those larger groups. Without the omitted counterfactual pair IDs, a normalized-state split alone cannot guarantee that both members stay together. Reconstruct pair groups with a reviewed matching procedure, or reserve that config until grouping is trustworthy. Do not infer that adjacent rows necessarily form pairs.

Create model-validation, calibration, and policy-validation partitions using group assignments. A starting allocation is 90%, 5%, 2.5%, and 2.5% of training groups. Check actual per-task counts, especially soft labels and score questions, before accepting the split. Leave all provided test splits sealed for the final release comparison.

Before using an external benchmark, check state text, source identifiers where available, and near-duplicate overlap against training. Normalized exact hashes are the first check, not the final check.

### 6.4 Data selection for the first run

Select 100,000 distinct decisions from the remaining training partition. These are decisions, not rows and not repeated training exposures.

Use this starting mixture, then adjust for the demo's actual task distribution:

| Source | Decisions in the first 100K | Purpose |
| --- | ---: | --- |
| `default` | 55,000 | Broad language understanding and task coverage |
| `hard_50k` | 20,000 | Confusable alternatives, exceptions, missing evidence |
| `skills` | 15,000 | Definition following, soft targets, varied instructions |
| `numeric_temporal` | 5,000 | Stress-test weaknesses rather than depend on arithmetic mastery |
| `counterfactual_15k` | 5,000 | Sensitivity to changes that should flip labels |

If pair groups are unavailable for the last source, use an equivalent amount of audited procedural counterfactual data with known parent IDs. This is preferable to guessing group membership.

Within each source, cap overrepresented instruction templates and sample by domain and task type. Keep the first mixture broadly similar to the intended product distribution; do not force every class to uniform frequency. Report the realized mixture after filtering, because desired quotas are not a claim that every filtered pool can supply them.

The dataset already contains enough examples for an informative pilot. Collect a fresh, independently labeled support-routing evaluation set now, before looking at model failures on it.

### 6.5 Data and model release rights

BEV's Hub metadata uses `license: unknown` for a mixed-source release rather than asserting one blanket license. The released BEV model is non-commercial. Keep upstream attribution and a source-rights manifest for any public model or redistributed data. Synthetic wording does not remove obligations attached to the source state. [BEV dataset card](https://huggingface.co/datasets/avbiswas/bev-decision), [BEV model license](https://huggingface.co/avbiswas/bev-decider-0.4B).

This report does not audit the terms of every upstream source. For an eventual commercial release, select sources with verified suitable terms before the final training run. That limitation affects release planning, not the technical usefulness of a private research pilot.

## 7. Training size and the Kaggle budget

### 7.1 What "optimal size" can mean here

You are adapting an already pretrained model. A pretraining rule that links billions of tokens to parameter count is not a decision-fine-tuning data requirement. Chinchilla studies compute allocation for pretraining; use learning curves for this task instead. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

Use three sizes, with the same grouped validation sets and evaluation protocol:

| Stage | Distinct decisions | Training exposure | Question answered |
| --- | ---: | --- | --- |
| Smoke test | 128 to 512 | Deliberately overfit a small set | Does the pipeline learn and save correctly? |
| Pilot | 20,000 | One pass, then inspect | Is the architecture worth scaling? |
| First serious model | 100,000 | One pass; a second only if validation supports it | Does diversity improve useful performance? |
| Expansion | 200,000 to 300,000 | One pass on a larger audited mixture | Does more data beat more epochs? |
| Full-data candidate | Remaining eligible part of 422,551 | Budget-driven | Is the last increment worth its GPU hours? |

Keep distinct questions, distinct source groups, tokens processed, and optimizer steps separate in the logs. A million exposures to repeated paraphrases are not a million independent examples.

At an effective batch of 64 decisions, one pass over 100K is approximately 1,563 optimizer steps. Two passes are approximately 3,125 steps. One pass over the entire unfiltered training set is approximately 6,603 steps. Group holdouts and length exclusions reduce the eligible count.

Choose the smallest model/data setting that meets the target. As a decision rule, require an expansion to improve a fresh or untouched validation metric, not just training loss. A roughly one-point accuracy improvement or a useful NLL/risk-coverage improvement is worth examining; compare it with confidence intervals and added training cost before calling it real.

### 7.2 Kaggle constraints

Kaggle currently documents P100 and dual-T4 GPU environments, a 12-hour CPU/GPU session limit, and 20 GB of persisted working disk. Its GPU guide describes a weekly quota around 30 hours that varies with resources and demand. Verify the actual allowance in your account before scheduling. [Kaggle notebook specifications](https://www.kaggle.com/docs/notebooks), [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).

Plan around one GPU initially. Two T4s do not become one shared larger-memory GPU. Distributed data parallelism keeps a model replica on each device and adds setup and communication cost. Benchmark it only after a single-device run is reliable. Check how the account charges quota for the selected accelerator; do not assume two devices provide free throughput.

Use fp16 autocast and a gradient scaler on the available T4 or P100 hardware unless the runtime reports native support for the intended bf16 path. Keep trainable adapter weights, optimizer state, the decision head, and loss calculations in fp32 where practical. Loading a checkpoint labeled bf16 does not establish native bf16 execution on your GPU.

Do not assume a particular FlashAttention installation works on these devices with an arbitrary 4D branch mask. Correct eager attention is the reference. An SDPA path that passes symmetry and leakage checks is the first optimization candidate.

### 7.3 Starting optimization settings

These are proposed starting values, not experimentally established optima:

| Setting | First serious run | Why |
| --- | --- | --- |
| Backbone | All 28 Qwen3-0.6B layers | Avoid deleting pretrained capacity before measuring it |
| Adapted layers | Indices 16 to 27 | Upper 12 layers offer a small adaptation budget |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Adapt question/option attention and readouts |
| LoRA rank and alpha | `r=8`, `alpha=16` | Low-cost initial capacity |
| LoRA dropout | 0 initially | Simpler repeatable baseline |
| Decision head | Width 256, one set-attention block | Small symmetric comparison model |
| Optimizer | AdamW | Standard baseline |
| Adapter learning rate | `1e-4` | Conservative initial adaptation |
| Head learning rate | `3e-4` | Newly initialized weights need more movement |
| Weight decay | `0.01`, excluding biases and normalization | Starting regularization |
| Schedule | 3% warmup, then cosine decay | Avoid large initial updates |
| Gradient clipping | Norm 1.0 | Limit unstable updates |
| Effective batch | 64 decisions | Compare runs at a stable update size |
| Microbatch | Try 2, then 4 or 8 after profiling | Actual mask and activation memory determine it |
| Total sequence cap | 1,024 tokens initially | Covers most measured inputs |
| Option-content cap | 64 tokens initially | Exclude rather than silently clip longer cases |
| Gradient checkpointing | Non-reentrant if supported and tested | Reduce activations without losing adapter gradients |
| Training KV cache | Disabled | No autoregressive generation or inference cache needed |
| Validation | About every 250 optimizer steps | Enough feedback without evaluating constantly |
| Checkpoints | Best, latest, and a recoverable periodic snapshot | Fit the session and disk constraints |

LoRA reduces the updated weights and optimizer memory, not the cost of all backbone computation. QLoRA reduces frozen-weight memory further but adds quantization complexity. Start with ordinary LoRA for a 0.6B model; use QLoRA only if measured memory prevents the chosen sequence length or batch. [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [PEFT LoRA configuration](https://huggingface.co/docs/peft/en/package_reference/lora).

Qwen's explicit head dimension matters for parameter counting. Its query projection output is `16 × 128 = 2048`, rather than 1024. For the four attention projections, rank-8 LoRA adds:

$$
8[(1024+2048)+(1024+1024)+(1024+1024)+(2048+1024)]
=81{,}920
$$

parameters per layer, or 983,040 over 12 layers. Confirm those shapes and the printed trainable parameter count before training.

Begin with a brief head-only phase of roughly 100 to 200 optimizer steps on the pilot, then enable the adapters. If it adds cost without a validation benefit, remove that phase from the full run. A head-only baseline also shows whether the pretrained representations already support the task.

With frozen lower layers and checkpointing, verify that adapter gradients are present. A run that logs a loss while accidentally training only the head is a plausible integration bug. When introducing a PEFT wrapper, confirm the forward signature still accepts all custom masks and readout indices.

### 7.4 Memory and throughput estimates

At roughly 0.6 billion parameters, 16-bit backbone weight storage is about 1.2 GB before runtime overhead. Full fine-tuning with ordinary mixed-precision Adam can require on the order of 16 bytes per parameter for weights, gradients, master weights, and optimizer state, roughly 9.6 GB before activations. These are storage estimates, not measured peaks.

For `B=4` and `L=1024`, a `[B,1,L,L]` fp16 mask alone uses about 8 MiB. At `L=2048`, it uses about 32 MiB. Eager attention scores can scale as `[B,heads,L,L]` and consume much more memory per active layer, especially during training.

Profile 50 representative optimizer steps after warmup, including forward, backward, accumulation, and optimizer updates. Measure padding, peak allocated memory, examples per second, and padded tokens per second. Use that rate to plan the run:

$$
\text{training hours}\approx
\frac{\text{decision exposures}}{\text{measured decisions per second}\times3600}.
$$

The following is a sensitivity calculation, not a Kaggle speed claim:

| Measured end-to-end rate | One pass over 100K | One pass over all 422,551 |
| --- | ---: | ---: |
| 2 decisions per second | 13.9 hours | 58.7 hours |
| 5 decisions per second | 5.6 hours | 23.5 hours |
| 10 decisions per second | 2.8 hours | 11.7 hours |

Add validation, checkpointing, downloads, and interruption allowance. Split long runs into resumable sessions. Save optimizer, scheduler, scaler, RNG, sampler progress, the split manifest, head, and adapter. Restore them before consuming more quota.

Spend CPU-only time on parsing, validation, splits, synthetic generation, and tokenization. Enable GPUs only for model work. Attach versioned data and weight artifacts to the Kaggle notebook when internet access is unavailable.

For a planning week with about 30 usable GPU hours, reserve roughly 2 for correctness/profiling, 4 for pilots, 12 for the main run, 6 for the most useful ablation, and 6 for calibration, evaluation, and recovery. Replace this allocation with measured costs. If the main run is slow, reduce the experiment count instead of leaving all experiments incomplete.

## 8. Synthetic data that adds information

### 8.1 The weakness in a rephrase-only plan

Your proposed QA-to-choices pipeline is useful, but a rephrased question usually adds less information than a new rule, a difficult distractor, or a changed decision boundary. It also creates label errors when a rephrase changes a qualifier.

Some QA answers do not naturally define one mutually exclusive correct option. Multiple paraphrases of the same correct answer are not valid competing alternatives in a single-label task. If several candidates are genuinely acceptable, either revise the question, remove the ambiguity, or use a justified soft target.

Option descriptions must be individually meaningful. Avoid candidates such as "the previous one", "both of the above", and "option B" unless those references are rewritten into explicit content. Such references make presentation order part of the semantics.

### 8.2 The proposed generation pipeline

1. Select an allowed source state and its known answer or executable rule. Assign a stable `parent_id` before generation.
2. Assign the parent to train, validation, or test before creating any variations.
3. Define exactly what the decision means and what evidence supports the answer.
4. Ask the generator for candidate descriptions, hard negatives, and at most one or two useful paraphrases.
5. Use code to validate the schema, label membership, candidate count, token budget, duplicate descriptions, and source references.
6. Verify the answer independently. Prefer an executable rule, source annotation, or program execution. A second model's agreement is a screen, not proof.
7. Generate matched counterfactuals where a specific state or rule change flips the correct answer.
8. Deduplicate and keep every descendant of a parent in the same partition.
9. Review a stratified human sample, then add only examples that target measured weaknesses.

A generation record should retain `parent_id`, source, source revision, rights information, task type, state, instructions, candidate IDs, descriptions, label or label distribution, verification method, generator identity, and prompt version. Preserve this metadata outside the model text.

### 8.3 A concrete generator brief

```text
You receive a source passage, a question, a verified answer, and evidence.
Return JSON with state, instructions, options, correct_id, and evidence_spans.

Preserve every source fact. Do not add facts that decide the answer.
Create 4 to 8 individually meaningful options.
Exactly one option must satisfy the question under the supplied evidence.
Use distractors that are plausible but fail for a specific stated reason.
Match style and length across correct and incorrect options.
Do not refer to option order, letters, or other options.
Keep output IDs opaque and outside the option descriptions.
Retain negation, dates, thresholds, units, exceptions, and uncertainty.
If the source cannot support one answer, return a rejection reason.
```

Do not show the known answer to the independent verifier. Otherwise the check becomes an exercise in confirming the generator's conclusion.

For verifiable procedural tasks, generate a structured world first, calculate the label, then render the world into text. Check that the rendered text still states the decisive values. This is particularly useful for thresholds, access rules, and dates, although the final application should still use deterministic code for exact calculations when possible.

### 8.4 First augmentation priorities

After the 20K pilot, select at most 10K to 20K new parent cases targeted at failures. Prioritize:

- Confusable routing descriptions whose distinctions appear only in the definitions.
- Opaque-ID renaming and changed definitions with a corresponding changed label.
- Rule exceptions and negation that override a common default.
- Missing evidence, with an explicit unknown or review outcome.
- Counterfactual state pairs and policy changes that require opposite decisions.
- Longer and shorter distractors, mixed candidate counts, and neutralized stylistic cues.
- Untrusted instructions embedded in state text.

Audit generated length, wording, and answer-position statistics. In a truly equivariant model, answer position cannot be a structural shortcut, but correct answers can still have distinctive phrasing or length.

Do not add a permutation-consistency loss by default. Exact structural symmetry should make it redundant at evaluation. If a meaningful discrepancy appears, fix preprocessing or the attention implementation before teaching the network to approximate a property it should already possess.

## 9. Evaluation that can support a strong claim

### 9.1 The baseline ladder

Run the following comparisons in priority order:

| Baseline or ablation | What it establishes |
| --- | --- |
| Majority or task-prior predictor | Whether a dataset shortcut explains apparent progress |
| Frozen Qwen plus the bilinear decision head | What pretrained representations already provide |
| Ordinary causal serialization plus the same shared head | How much ordinary option order affects the model |
| Reset positions with ordinary causality | Whether position reset alone leaves the diagnosed bug |
| Branch mask plus reset positions, bilinear-only head | The minimal symmetry-correct architecture |
| Same construction plus one set-attention block | Whether explicit option comparison improves accuracy |
| Same construction plus LoRA | The value of backbone adaptation |
| BEV or another released small decision model | A realistic open-model reference |

You do not need full-scale training for every row. Use the same 20K pilot pool for the architecture ablations, then scale only the winning two candidates. Track equal training exposure and, separately, equal compute comparisons.

A prompt-based Qwen baseline can score answer-label token logits in one prefill when the labels are unambiguous single tokens. This is useful but not identical to scoring description embeddings. Verify tokenizer behavior and do not compare it as if a multi-token label had one token probability.

### 9.2 Correctness and uncertainty metrics

For every model, save per-example probabilities and report:

- Choice accuracy, macro-F1 for recurring semantic classes where meaningful, and accuracy by candidate count.
- Noul accuracy at a stated threshold, Brier score, and AUROC only where both labels occur.
- Score argmax accuracy, expected-rank mean absolute error, and ranked probability score.
- Negative log-likelihood and top-label calibration error, with the binning rule and sample count stated.
- Binary probability calibration for Noul, rather than only selected-label confidence.
- Risk versus coverage when uncertain predictions are sent for review.
- Results by domain, input length, hard-negative family, and definition-following stress case.

Define multiclass Brier as the mean over examples of `Σᵢ(pᵢ−tᵢ)²`, and keep that convention fixed. Its scale and baseline depend on candidate count, so include count-stratified results. For hard-label Noul, state whether you use scalar binary Brier or the two-class sum.

Soft-target examples need their own distribution-quality report. Hard accuracy against an argmax label loses the information that made those examples useful.

Use group-aware bootstrap intervals when several questions share a state or source family. Small external suites can make a one-question change look like a large percentage improvement. Compare the same states and candidate definitions under the same grading rule.

### 9.3 The symmetry and leakage test suite

The architecture must pass these before a long training run:

1. Enumerate all permutations for a small set of 2 to 5 options with unequal lengths. For larger sets, test at least 20 random permutations per case.
2. Undo each permutation by stable candidate ID and compare logits and probabilities.
3. Repeat in fp32 and the actual deployment precision, with eager and the chosen optimized backend.
4. Repeat with different padding, batch companions, and batch ordering.
5. Change one option's text. Other option readouts must remain unchanged before the set head, while the summary and final scores may change.
6. Append an isolated question to a batch. Existing question results must stay within the established numerical tolerance.
7. Check that every invalid candidate probability is zero, all real probabilities are finite, and valid probabilities sum to one.
8. Rename opaque candidate IDs. Predictions mapped by descriptions must agree, except for an explicitly reported tie selection.
9. Inspect target alignment after every shuffle, especially list-shaped soft targets and ordinal ranks.

Start with an fp32 comparison tolerance around `1e-5` absolute and `1e-5` relative, and an fp16 probability-drift investigation threshold around `1e-3`. These are proposed checks, not universal numerical bounds. Measure their distribution on the actual hardware. A failed check needs investigation rather than a silent increase in tolerance.

Report maximum and percentile drift, exact-tie count, and changed winners as a function of the original top-two margin. Close margins explain some numerical flips; they do not justify large score movement.

Architectural symmetry is broader than test coverage, but tests are still required to establish that the implementation matches the proof.

### 9.4 Independent benchmarks and the demo set

Use provided BEV test splits only for final in-mixture evaluation. Add a fresh source-disjoint support-routing set with about 1,000 states, including normal, ambiguous, missing-evidence, and adversarial cases.

Public Jev-oriented suites such as sysone-bench are useful additional comparisons. Pin the release, gold-label version, manifest, and adapters. The current repository distinguishes historical sealed results from corrected labels and later panels, so copying a leaderboard number without its protocol is unreliable. [sysone-bench](https://github.com/instax-dutta/sysone-bench).

Use JevBench public tasks only with the matching public grading protocol, and keep public-subset claims distinct from full-suite claims. The BEV card's cited benchmark results are context, not targets guaranteed by this design. Existing benchmark families may overlap the training mixture. If a benchmark cannot be shown source-disjoint, label it accordingly rather than calling it independent.

To claim "best small decision model," define the eligible models, parameter range, identical evaluation inputs, resource limits, and uncertainty metrics. Without that scope, "best" is not a research result.

## 10. Efficiency after correctness

### 10.1 Compare packing with separate question and option pairs

Packed branches avoid encoding the prefix once per candidate and enable a shared summary. They also create a long sequence and an unusual attention mask. Dense kernels may spend time on blocked token pairs.

A simple correctness reference is to encode `[Q, Cᵢ]` as an independent ordinary causal sequence for each option with shared weights, then combine the readouts through the same position-free head. The backbone option representations should match the branch construction within numerical tolerance when serialization and positions agree. A learned head summary can replace the packed global summary for this alternative.

For the independent-pair alternative, repeated prefix tokens can make it more expensive, but standard kernels and short sequences can compensate. Benchmark both for representative `(prefix length, option count, option length)` combinations. Do not infer speed from the number of Python calls alone.

### 10.2 Shared-prefix caching is a later optimization

Because the prefix cannot attend to choices, its layer-by-layer key/value states are reusable across option branches for the same state and question at inference. Use cache fan-out only after a cached path matches uncached results.

With 28 layers, 8 KV heads, head dimension 128, and 16-bit caches, a 1,024-token prefix uses approximately:

$$
2\times28\times1024\times8\times128\times2
=117{,}440{,}512\text{ bytes},
$$

about 112 MiB, before cache objects and other tensors. Repeated cache copies across candidates can erase the benefit. Shared storage or appropriate batching matters.

Do not detach an adapted prefix cache during training as if it were an inference optimization; doing so changes gradient flow. Start training with the simpler packed or independent-pair computation.

### 10.3 Dense masks are not sparse execution

The intended branch attention has useful work closer to:

$$
O\left(P^2+\sum_i(P\ell_i+\ell_i^2)+aL\right).
$$

A dense implementation can still do attention work proportional to `L²`, where `L = P + Σℓᵢ + a`. The masked edges are logically absent but may still cost matrix multiplication and memory.

FlexAttention or a specialized block implementation could exploit the pattern later. Compiler time, GPU support, and maintenance cost need measurement. Neither custom sparse kernels nor a new dependency belongs in the first Kaggle run.

### 10.4 Compression choices

Keep all 28 layers until you have a quality baseline. Then compare a 20-layer variant at the same head and data budget. Dropping layers saves compute but may lose the semantics needed by difficult decisions. The BEV release demonstrates a 20-layer construction, not that the first 20 layers are optimal for every task.

Quantize only after measuring the unquantized model. Re-test probability calibration, order drift, and near-tie selection after quantization and after LoRA merging. Calibration is a property of the deployed model and precision, not just the pre-export checkpoint.

## 11. A demo that makes the claims testable

### 11.1 The first product scenario

Build a support-ticket decision panel. A user provides a ticket and a small set of queue definitions. One request evaluates:

- `choice`: which queue fits the definitions.
- `noul`: whether the ticket includes a deadline or time pressure.
- `score`: severity against an explicit ordered rubric.

Show probabilities, the selected typed result, and whether the application recommends review. Use original examples with labels written before model evaluation. Include failures and ambiguous cases in the demo suite.

### 11.2 The useful interactions

The demo should let a reviewer shuffle and drag choices, rename opaque IDs, edit definitions, and replace decisive facts in the ticket.

Shuffling tests the architectural claim. Renaming IDs tests input rendering. Editing a definition tests whether the model follows the rule. Changing a decisive fact tests whether the answer changes when it should.

Show per-key probability differences against the original ordering. Mark numerical ties and near-ties instead of forcing a reassuring stability indicator. Keep candidate colors or labels tied to identities, so the display does not create its own apparent order changes.

A separate length/error view should show why an input was rejected and the documented limit. Oversized state text must not disappear silently.

Do not generate an explanation and present it as the decision model's reasoning. A decision model can return its distribution without a textual rationale. If evidence highlighting is added later, evaluate that feature separately.

### 11.3 Deployment and latency measurement

Train on Kaggle, then export weights, tokenizer, configuration, temperatures, and split/evaluation manifests. Run the demo in a local service or a suitable inference host. Kaggle is the training environment, not a permanent public API host.

A minimal service needs validated request types, bounded sizes, the same serializer as training, model version information, and one typed response builder. Batch independent questions without allowing them to attend to one another. Multiple questions in one HTTP call do not imply free computation.

Measure warm model latency at batch size one and at the chosen question batch size. Report tokenization, network forward, response construction, cold start, and end-to-end latency separately. State GPU or CPU, precision, input token count, candidate count, warmup, and concurrency. Synchronize CUDA when timing GPU work.

Proposed first-demo goals are at least 95% accuracy among accepted routing decisions at 70% or more coverage on the fresh test set, and a responsive local interaction with short inputs. Those are engineering targets, not predictions. Set a numerical latency goal after the first hardware measurement. A 0.6B model cannot inherit TypeSafe's hosted-service speed claim.

For the accepted-decision goal, report the accepted count and a confidence interval. If the target is missed, increase review coverage or narrow the supported task and state the resulting coverage. Do not select a favorable test threshold after seeing the final answers.

## 12. The experiment sequence

| Order | Deliverable | Completion criterion | Decision after it |
| --- | --- | --- | --- |
| 1 | Frozen data and evaluation manifests | Stable hashes, grouped splits, a fresh demo set | Confirm the evaluation is usable |
| 2 | Encoder, mask, and bilinear head | Structural checks, finite outputs, correct target mapping | Fix any symmetry or leakage bug |
| 3 | Tiny training smoke test | Overfits a small clean set; save/reload reproduces results | Confirm gradients and checkpointing |
| 4 | Eager versus SDPA profile | Equivalent outputs and measured memory/rate | Pick the fastest correct backend |
| 5 | 20K pilot ablations | Same data and validation protocol | Choose bilinear-only or small comparison head |
| 6 | 100K LoRA run | Best checkpoint selected by untouched validation | Establish the first serious baseline |
| 7 | Calibration and policy thresholds | Separate fitting and threshold partitions | Freeze the deployable candidate |
| 8 | Targeted augmentation or 200K expansion | Better independent validation per GPU hour | Spend remaining quota on the stronger option |
| 9 | Final sealed evaluation | Per-example outputs, group intervals, drift and latency | Accept or revise the supported scope |
| 10 | Demo and model card | Fresh examples, failure cases, limits, reproducible artifact | Make the results reviewable |

The first implementation can stay small: one dataset preparation path, one serializer/collator, one backbone-plus-head module, one training notebook, one evaluation path, and one demo service. A generic framework, agent orchestration, multiple interchangeable trainers, and reinforcement learning would add work before resolving the core question.

Keep a run manifest with base and dataset revisions, all package versions, source groups, lengths, head dimensions, LoRA target layers, seeds, hyperparameters, precision, best step, trained tokens, GPU hours, calibration version, and deployment checksums. The saved head is part of the model. Saving only LoRA would lose the newly learned decision function.

## 13. A runnable structural self-check

This small Python check verifies the mask and position specification across all six permutations of three unequal-length options. It is a graph check, not a Qwen forward-pass test. The check was executed successfully while preparing this report.

```python
from itertools import permutations

def layout(order):
	lengths = {"a": 2, "b": 4, "c": 1}
	prefix_length = 3
	tokens = [("q", "", t) for t in range(prefix_length)]
	positions = list(range(prefix_length))
	for key in order:
		tokens += [("c", key, t) for t in range(lengths[key])]
		positions += list(range(prefix_length, prefix_length + lengths[key]))
	tokens += [("a", "", t) for t in range(2)]
	positions += list(range(prefix_length + max(lengths.values()),
		prefix_length + max(lengths.values()) + 2))
	return tokens, positions

def allowed(query, key):
	q_kind, q_id, q_t = query
	k_kind, k_id, k_t = key
	if q_kind == "q":
		return k_kind == "q" and k_t <= q_t
	if q_kind == "c":
		return k_kind == "q" or (
			k_kind == "c" and q_id == k_id and k_t <= q_t)
	return k_kind != "a" or k_t <= q_t

base_tokens, base_positions = layout(("a", "b", "c"))
position_by_token = dict(zip(base_tokens, base_positions))
for order in permutations(("a", "b", "c")):
	tokens, positions = layout(order)
	assert dict(zip(tokens, positions)) == position_by_token
	index = {token: i for i, token in enumerate(tokens)}
	base_index = {token: i for i, token in enumerate(base_tokens)}
	mask = [[allowed(q, k) for k in tokens] for q in tokens]
	base_mask = [[allowed(q, k) for k in base_tokens] for q in base_tokens]
	for q in base_tokens:
		assert any(mask[index[q]])
		for k in base_tokens:
			assert mask[index[q]][index[k]] == base_mask[base_index[q]][base_index[k]]
			if q[0] == "c" and k[0] == "c" and q[1] != k[1]:
				assert not mask[index[q]][index[k]]
			if q[0] == "q" and k[0] != "q":
				assert not mask[index[q]][index[k]]
	assert mask[index[("a", "", 1)]][index[("c", "b", 3)]]
	assert not mask[index[("a", "", 0)]][index[("a", "", 1)]]
	assert not mask[index[("c", "a", 0)]][index[("c", "a", 1)]]
	assert mask[index[("c", "a", 0)]][index[("q", "", 2)]]
print("Structural checks passed for all six option permutations.")
```

The real implementation still needs the tensor, attention-backend, gradient, padding, probability, and checkpoint tests in section 9.3. Passing this check alone cannot validate an end-to-end model.

## 14. Evidence, assumptions, and remaining work

The report directly inspected your three diagrams, the current official Jev contract, BEV's encoder and head code, Qwen's configuration, and primary papers. It also performed the complete Parquet structural audit, the soft-target value audit, and the seeded token-length sample described above.

Pinned research inputs:

| Input | Revision or inspected version |
| --- | --- |
| BEV dataset | `11ed9b5b879e7f9524e7c89c1037d498c54206f0` |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| BEV inference repository | `aa5d861f256b57f32794715de2559cd5a7b68a30` |
| Qwen and SDPA implementation source | Transformers `v4.57.1` |
| Set-LLM | arXiv `2505.15433v1` |
| Label-bias preprint | arXiv `2610.02586v1` |

The report did not train a model, run Qwen forwards, reproduce published benchmarks, measure Kaggle speed or memory, verify this account's quota, recover missing provenance, or audit every upstream license. Numerical settings and demo success criteria are proposed experiments. Published benchmark figures are attributed to their authors.

The first work that resolves the largest uncertainty is a correct encoder/head implementation followed by the 20K pilot and a real GPU profile. After that, the choice between more data, a wider head, more adapter capacity, and a shorter backbone can be based on measured improvement rather than architectural preference.
