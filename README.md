# XIOS

A chat-model architecture built on one idea: **a model is big for two unrelated reasons, and we conflate them.**

Parameters store *knowledge* — facts, which genuinely need bits. Parameters also perform *computation* — reasoning, which needs sequential depth. We buy both by making one matrix stack wider and deeper. XIOS separates them, and each half then gets to be cheap in its own way.

This is a research architecture with a working implementation, not a finished model. It ships with the experiment designed to prove it wrong.

---

## The three mechanisms

### 1. Depth is allocated at runtime, under a budget

Every transformer in use today spends identical compute on every token. Predicting the `the` in "one of the" costs exactly what the final step of a proof costs. That isn't a tuning inefficiency — depth is baked into the weight file.

In XIOS a small core is **looped**, a token-dependent number of times, under a fixed global budget. Easy tokens exit after one pass; hard tokens loop twenty or thirty times. Mean FLOPs per token stay flat. What changes is that hard tokens receive sequential compute far beyond what the parameter count would normally buy, because the loop reuses the same weights.

```
tokens → embed → prelude (dense, once)
               → core  (shared weights, 1..N iterations)   ← all the reasoning
               → coda  (dense, once) → logits
```

Halting is geometric ACT, but the decision depends **only on that token's own hidden state**, so it is exactly causal. No auxiliary predictor is needed at inference — unlike top-k depth routing (Mixture-of-Depths and relatives), where a token's fate depends on other tokens' scores, which leaks future information during training and needs a separate causal predictor bolted on for decoding. `tests/test_ponder.py` verifies the causality directly: rewriting positions 25–39 leaves the depths chosen for positions 0–24 bit-identical.

Measured allocation on an untrained nano core with a data-dependent halt head — the active set falls off geometrically, which is where the reinvested compute comes from:

```
depth histogram:  {2:11, 3:37, 4:27, 5:15, 6:7, 7:4, 8:27}   mean 4.70, ceiling 8
active fraction:  1.00 → 1.00 → 0.91 → 0.63 → 0.41 → 0.30 → 0.24 → 0.21
```

### 2. Each depth level keeps its own memory timeline

This is the part I believe is genuinely new.

The recurrent mixers keep a **separate state per iteration index**. State `S[i]` is the memory of *everything still being thought about at depth i*. Tokens drop out as depth increases, so `S[0]` sees the full token stream while `S[12]` sees only the handful of tokens hard enough to reach depth 12.

Two things fall out of that:

- **Deep computation runs over an automatically compressed, salient history.** Context stays full at the bottom of the stack and compute falls off geometrically toward the top. Depth selection and history compression become the same mechanism, for free.
- **Thinking persists across tokens.** Those states carry across generation steps, not just within a token. When token 40 iterates twelve times, its twelve iterations write into twelve timelines that token 41 then reads. Deliberation accumulates into the sequence instead of being discarded at the token boundary — what a chain of thought buys in token space, bought in latent space at no token cost.

A Universal Transformer loops but discards the loop state per token; latent-recurrence work does the same. Here the loop state *is* the sequence state.

Making this correct required care in two places, both verified: the linear recurrence is masked rather than gathered (exact, because the recurrence is linear in its writes), and the decay across skipped tokens is replayed at the base rate so a head's half-life stays calibrated in *real* token units rather than stretching silently at depth.

### 3. Knowledge lives on the SSD, not in the weights

An FFN is already a key-value memory — the rows of `W_in` are keys, the rows of `W_out` are values. The problem is that it scores *every* key on *every* token, so knowledge capacity costs RAM and FLOPs in lockstep.

XIOS makes the lookup explicit: keys factorised into two small codebooks (scoring `N` slots costs `O(√N)`), values in a memory-mapped int8 file, only the top 32 ever touched. The store is partitioned per head, and the physical layout is a seriated Z-order that costs `O(√N)` to describe rather than `O(N)`.

Result — **knowledge stops costing RAM**:

| preset | resident params | RAM (int4) | memory slots | disk store | total download | effective depth |
|---|---|---|---|---|---|---|
| nano | 14 M | 0.01 GB | — | — | 0.01 GB | 11 … 26 |
| micro | 103 M | 0.07 GB | — | — | 0.07 GB | 24 … 84 |
| small | 400 M | 0.22 GB | 8.4 M | 2.18 GB | **2.4 GB** | 52 … 196 |
| base | 884 M | 0.49 GB | 33.6 M | 8.72 GB | **9.2 GB** | 86 … 326 |

`base` needs **under half a gigabyte of RAM**, ships as a 9 GB download, and applies up to 326 block-applications of sequential computation from 16 stored blocks — a 20× leverage of effective over stored depth. The 33.6 M-slot knowledge store is 4.9× the size of the resident fp16 weights and none of it is ever in RAM.

Verified round trip (`tests/test_memory_disk.py`): export to int8 + per-slot scales, memory-map it back, and the disk backend agrees with the in-RAM one to **0.6 %** relative error at **68 bytes per slot**. The seriated layout survives the round trip, so addresses and values stay in sync.

---

## What's measured, and what isn't

Every claim below is produced by a test in `tests/`. I'd rather hand you a short list of verified things than a long list of asserted ones.

### Verified

| claim | result |
|---|---|
| Chunk-parallel recurrence ≡ true sequential recurrence | rel. err **7.5e-7** |
| Prefill-then-decode ≡ full parallel forward | rel. err **4.7e-7** |
| Depth sparsification is exact — inactive tokens cannot reach active ones through the mixer *or* the depthwise conv | max diff **0.0** |
| Gapped decode ≡ gapped parallel (per-depth timelines agree) | rel. err **4.0e-7** |
| **Full model: parallel forward ≡ incremental decode** | **100 %** argmax agreement, 5.8e-7 logit err |
| Halting is causal (future tokens cannot change earlier depths) | **25/25** positions identical |
| Depth allocation is heterogeneous, active set shrinks with depth | 1.00 → **0.21** |
| Budget curriculum is monotone and lands on target | 10.2 → 3.0 |
| Lagrangian price rises over budget, decays under it | 0 → **1.0** → 0 |
| Budget term stays a shaping force, never dominates the LM loss | 0.94 at 1.9× budget |
| **XIOS matches a dense baseline on a task the baseline solves** | **100 %** vs 100 % |
| Budgeted quantity tracks iterations actually run | 5.29 vs 5.54 (**4.5 %**) |
| `Σ Rᵢ` demonstrably under-counts real compute | **2.1×** low |
| `hop` single-cycle labels are distinct for every k < width | verified, k ≥ width refused |
| XIOS in-distribution at hardest trained difficulty | **100 %** vs 93.4 % |
| **XIOS past the dense model's depth cliff (3-8 compositions)** | **100 %** vs 2.7-5.5 % |
| XIOS at an unseen composition depth | **48.0 %** vs 4.7 % (chance 4.2 %) |
| **Result holds at matched FLOPs vs a 2.8x larger dense model** | **100 %** vs 2.3-5.9 %, at 1.00x compute |
| Memory addressing is a bijection over all slots, both layouts | exact |
| Layout table is O(√N), not O(N) | **8.2 KB** vs 0.5 MB |
| SSD page reads, seriated Z-order vs row-major | **1.83×** fewer |
| int4 compression (resident weights, vs fp16) | **3.28×** (small), **3.42×** (base) |
| int8 mmap store ≡ in-RAM store | rel. err **0.6 %**, 68 B/slot |
| Seriated layout survives export/attach | rel. err **0.6 %** |

The decode-consistency test matters most: with per-depth memory timelines it's the invariant most likely to break silently, and a model that trains under one set of semantics and generates under another is worthless.

### Wrong ideas, found by measuring

Eleven things I asserted turned out to be false or insufficient. They're listed because the corrections are the actual content — two were outright bugs in the central mechanism, one invalidated the benchmark, and one meant the headline claim was not being enforced at all. Every one was caught by measuring rather than reasoning.

**The loop kills itself.** With the obvious budget loss — penalise only *exceeding* the target — mean depth collapsed from 6.8 to **1.05 in fifty steps**, reducing XIOS to a shallow stack and defeating the whole architecture. Early in training the later iterations are untrained and therefore useless, so halting immediately is genuinely optimal; but once the controller halts at depth 1, those iterations never receive gradient and can never become useful. Fixed by two changes, both required: a **two-sided** budget loss (mean depth pinned *at* the budget, so compute freed from easy tokens must be spent rather than evaporating — this is what makes the scheme reallocation rather than mere reduction), and an **annealed** budget (start generous, tighten to the real target). After the fix, depth held at 2.6–3.2 and — with no term in the loss asking for it — **rose monotonically with problem difficulty (2.62 → 3.23)** within 150 training steps.

**Morton layout, wrong twice.** I claimed Z-ordering the product-key grid would collapse retrieval into contiguous runs. Measured **0.85×** — worse than doing nothing, because retrieval selects a rectangle of *scattered* codes, not a contiguous 2-D box. Second attempt sorted each axis by its leading principal component first: still worse (**0.78×**), because trained codebooks lie on a *curved* manifold where PC1 is many-to-one and actively scrambles the order (correlation with true order: 0.73). The version that works needed two further corrections — **per-head partitioning** of the store (one address table cannot localise several heads with unrelated key geometries) and **spectral seriation** via the Fiedler vector instead of PC1. Result **1.83×**, and the seriation recovers essentially the same locality as an index-sorted oracle (43.7 vs 44.5 pages), meaning it fully recovers the manifold ordering.

**AWQ's published exponent hurts here.** Fixing the activation-scaling exponent at 0.5 made int4 error *worse* (0.0708 → 0.0820). At group-64 the grouped scales already absorb most of the outlier problem, and scaling one channel up widens its group's range and degrades every other channel sharing it. Replaced with a search over the exponent, which correctly selects **α = 0** here — so the feature is now monotonically safe rather than a liability. With scaling off, the low-rank error corrector then earns its bytes: **0.0708 → 0.0530** at rank 64, a 25 % error reduction.

**The looped core had two real bugs, and a trivial task found both.** Before measuring whether XIOS is *better*, it has to clear the floor: match a dense baseline on a task the baseline solves outright. `tests/test_learns.py` does exactly that — copy one character out of a string — and XIOS failed it at **64.8 %** against the baseline's **100 %**.

Two independent causes, each worth a fix:

1. *Renormalising the residual stream every iteration.* The loop body computed `RMSNorm(h) + inject(x0)`, which throws away accumulated magnitude on every pass. The blocks are pre-norm and already normalise their own inputs, so this norm was not merely redundant — it destroyed information the loop needed to carry across iterations. Removing it: **64.8 % → 84.0 %**.

2. *Textbook ACT output-mixing.* Standard adaptive computation accumulates `y = Σ Rᵢ pᵢ hᵢ`, blending the hidden state from *every* iteration into the output — which necessarily mixes under-computed intermediate states into a finished answer, blurring it. Replaced with **update-gating**: the halt probability now scales how much each iteration may *write* to the residual stream, so a halted token simply stops updating and its state is whatever it had reached, never a smear across depths. Still fully differentiable (gradient reaches the halt head through the gate that scales each write), and in the hard limit it is exactly "stop computing this token". **84.0 % → 100.0 %**, matching the baseline.

That second fix is the more interesting one: the standard ACT formulation is actively harmful in a deep looped core, and the fix is a strictly better formulation of the same idea.

**The compute budget was pricing the wrong number — so compute-neutrality was never enforced.** This is the most consequential mistake in the project, and it hid behind a log line that looked fine.

The halting maths gives `E[iterations] = Σᵢ Rᵢ`, where `Rᵢ` is the probability a token is still running. That identity holds when halting is *sampled*. Execution does not sample: it runs deterministically until `R` drops below a cutoff `eps`, which for geometric halting takes about `ln(eps)/ln(1−p)` iterations. The two quantities are not the same number, and budgeting the wrong one is silent:

| | value |
|---|---|
| iterations actually run | **5.54** |
| `Σ Rᵢ` — what the budget was constraining | **2.59** |

A **2.1× under-count**. The budget was satisfied — λ sat at exactly 0.000 — while depth was pinned at 5.94 of a 6-iteration ceiling. *The central claim of the architecture, that freed compute is reallocated rather than compute simply growing, was not being enforced at any point.*

Fixed by budgeting a differentiable surrogate for the real iteration count, `Σᵢ σ((Rᵢ − eps)/τ)`, which tracks actual iterations to within 4.5 % (5.29 vs 5.54) and stays differentiable in the halting probabilities. λ immediately began to engage (0.000 → 0.125 within 250 steps) where before it never moved.

Two lessons worth keeping. First, a constraint that is *satisfied* is not evidence it is *binding* — check that the multiplier ever leaves zero. Second, the log was reporting mean depth over all positions while the budget was computed over the loss-masked positions only; those can differ wildly, and reporting one while controlling the other is an efficient way to draw exactly the wrong conclusion. `PonderStats` now exposes `mean_depth`, `scored_depth` and `compute_depth` separately, and the trainer prints the first two side by side.

**A fixed penalty weight does not hold the budget.** With `budget_weight = 0.05` and a target depth of 3.0, mean depth settled at **5.68** and stayed there for 600 steps. The language modelling loss simply outbids a constant penalty — the model is happy to pay it in exchange for compute. Raising the constant by hand is guesswork that has to be redone for every model size, task and learning rate. So the budget is now treated as a **constraint** rather than a preference, priced by dual ascent:

```
lambda ← clamp(lambda + eta · (mean_depth − target), 0, lambda_max)
loss   += lambda · (mean_depth − target) + w · (mean_depth − target)²
```

Verified in `tests/test_ponder.py`: against a model insisting on depth 5.7, λ climbs until overspending costs 27.4 (versus an LM loss around 1.0), and decays back to 0 once the model comes in under budget. λ is also a useful readout on its own — it *is* the marginal price the model puts on one more iteration.

Worth noting what the stall tells us: given the choice, this model wanted nearly *twice* its budgeted depth and paid a real penalty to keep it. That's a point in favour of depth mattering, but it means the compute-neutrality has to be enforced, not hoped for.

**The depth benchmarks were measuring the wrong thing.** Three separate problems, all found by checking rather than assuming.

*No curriculum.* Training directly on 1–5-step instances, both models plateaued at 0.97 nats — almost exactly `log 7`, the entropy of guessing. A model that cannot do two steps gets every five-step instance wrong and receives no useful gradient. Difficulty now ramps over the first third of training.

*Width set past the model's reach.* With the original 12-entry lookup table, a dense 4-layer baseline never learned even a **single** hop, so the entire depth curve was noise. Measured, single-hop exact match after 700 steps:

| table entries | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| exact match | 100 % | 100 % | 31 % | 11 % | 6 % (chance 8 %) |

The lesson generalises: **verify the model can do one step before measuring how many it can chain.** Each task now has a *width* knob separate from its *depth* knob, `calibrate_width()` finds the largest width a given model actually learns, and the A/B harness prints a warning if neither model solves the one-step case — because in that regime the table above it means nothing. (RoPE theta was ruled out as a cause: sweeping it from 5×10⁵ down to 10² changed nothing.)

*A leaky generator, twice.* The pointer-chase task sampled arbitrary permutations, so fixed points like `a→a` made some instances solvable in zero hops regardless of their declared step count. Switching to derangements fixed the fixed points but not the deeper problem: **following one table `k` times lands at `k mod cycle_length`**, so difficulty was never monotone in `k` at all. The first completed A/B made this unmissable — exact match across steps 1–9 came out

```
100%   18%   19%   45%    0%   80%    0%   40%   13%
```

The 80 % at six steps is the model discovering that after six hops around a short cycle the answer is usually just the start node. A benchmark whose difficulty axis isn't monotone cannot measure depth, and **a flat depth curve measured on it means nothing** — which is precisely what that run reported. `hop` now builds a single n-cycle and refuses any chain that would wrap, at the cost of capping testable depth at `width − 1`. For deeper chains the right task is `perm`, where every step applies a *different* permutation and there is no wrap-around shortcut by construction.

The harness now carries three independent guards, each from a mistake above: it clamps the step range to what the task can label, warns if neither model solves the one-step case, and warns if accuracy *rises* with difficulty more than once — the signature of a shortcut.

**I reported a headline result that was a measurement artefact.** The depth-vs-difficulty correlation of +1.000 — the finding I was most pleased with — came from averaging depth over all sequence positions including padding, whose share varies with problem size. Corrected, the same run gives −0.256 and depth is flat. The warning sign was there and I walked past it: the *other* run gave −1.000, and a mechanism cannot be both perfectly correlated and perfectly anti-correlated with difficulty. Two contradictory perfect correlations mean the statistic is measuring something else. Reporting one number while controlling a different one (unmasked depth vs masked budget) is the same failure that hid the compute-budget bug above, and it bit twice in the same project.

**My own size reporting was misleading.** The memory's value slots were being counted as parameters, inflating `small` from 400 M to 2.5 B and making the compression ratio meaningless; a tied output head was being silently un-tied into a 100 MB fp16 copy during quantisation, erasing most of the saving; and the address table was O(slots) resident. All three are fixed, and `param_report()` now separates resident RAM from disk bytes, because conflating them is exactly how this kind of claim gets faked.

### Not established

- **That XIOS beats a normal transformer where it counts.** See the result below: the depth mechanism demonstrably works, but the accuracy payoff it was supposed to buy is not there yet.
- Quality at useful scale — everything here is verified at nano/micro.
- That the disk memory's latency is tolerable on real consumer SSDs under load. The IOPS reduction is measured; the wall-clock is not.
- The int4 dequant path is memory-optimal but not speed-optimal; it needs a fused kernel to convert the size win into a speed win.

---

## The constraint that actually matters: bandwidth, not size

This reframed the whole project, and it came from asking what physically
limits a local model rather than what the parameter count is.

**Decoding is memory-bound.** Generating one token from a dense model reads
*every weight* from DRAM, because each weight is used once and then not again
for that token. A 7B int4 model is 3.5 GB; at a laptop's ~12 GB/s that caps
you at roughly 3 tokens/second no matter how fast the chip does arithmetic.
Optimising FLOPs against that wall accomplishes nothing. This is the real
reason "you need a lab" feels true — and it is a *bandwidth* claim masquerading
as a size claim.

**A looped core is the one shape that beats it.** A dense stack must stream all
its weights per token. A looped core reuses the same weights `depth` times, so
if it fits in a cache tier it is fetched from DRAM **once** and every further
iteration is served on-die:

```
dense    DRAM bytes/token = all weights
looped   DRAM bytes/token = core (once) + prelude + coda + head
         cache traffic    = core x depth        (nearly free)
```

Depth becomes almost free in exactly the resource that is scarce. Measured
(`tests/test_bandwidth.py`): raising depth from 4 to 64 multiplies DRAM traffic
by **1.000x** while cache traffic goes 21 → 441 MB.

### This showed my own presets were built backwards

I had been scaling the core *wider* as presets grew, which pushes it out of
cache — and then every iteration re-reads it from DRAM, so depth multiplies
bandwidth instead of being free:

| preset | core | lives in | DRAM/token | vs 3B dense |
|---|---|---|---|---|
| nano | 10 MB | L2/L3 | 8 MB | 208x fewer bytes |
| micro | 36 MB | L3 | 61 MB | 27.8x fewer |
| small | 828 MB | DRAM | 878 MB | 1.9x |
| base | 2.4 GB | DRAM | 2,496 MB | **0.69x — slower than dense** |

`base` was worse than a conventional model on precisely the hardware it was
designed for. The design rule is the inverse of what I built: **hold the core
inside L2/L3 and spend everything else on iterations.** Width is a constraint,
not a knob. `xios.bandwidth.design_for_bandwidth()` now searches configs under
that constraint instead of me guessing, and it makes mechanism 3 load-bearing
rather than optional: with width capped, the disk memory is the only place
knowledge *can* live.

### The `flagship` preset, designed by that rule

| | |
|---|---|
| resident weights (int4) | **27 MB of RAM** |
| disk knowledge store | 8.72 GB (33.6 M slots, never resident) |
| total download | **8.75 GB** |
| block-applications / token | 196 average, 388 max, from 8 stored blocks |
| DRAM bytes / token | 30.8 MB vs 1,695 MB for 3B dense — **55x fewer** |
| bandwidth-bound rate | ~237 tok/s vs ~7 tok/s — **34x faster** |

27 MB resident. That runs on anything with a pulse. It is also ~8x the
sequential depth of a 32-layer dense model at ~30x the decode speed, which is
the combination the project was actually after.

Untrained, to be clear — this is a *design point* validated on bandwidth and
depth, not a trained model. But it is the first configuration here whose
resource profile is genuinely consistent with "flagship reasoning on bad
hardware", and it got there by obeying a measurement instead of scaling a
number up.

### Next bottleneck, and one idea that failed

With the body cached, the analysis flags that the **output projection is 54% of
all DRAM traffic** — 16.8 MB per token, and unlike the core it cannot be cached
because every token needs a different part of it. So the head is now the thing
to attack, not the model.

`xios/head.py` tries: cluster the vocabulary by row direction, score cheap
centroids, then read *only* the top clusters and score those rows exactly. On
the real trained demo head (vocab 2048) that gives **94.5% top-1 agreement at
10.3x fewer bytes**, with the approximation confined to *which* tokens are
considered — scores inside the candidate set are exact.

I also tried to make it **exact**, via a genuine upper bound on the best logit
a cluster can contain (`R_c * ||h|| * cos(max(0, angle(u_c,h) - theta_c))`),
visiting clusters in bound order and stopping once the best logit found beats
the next bound. It is branch-and-bound over a cone decomposition, it is
provably correct, and **it does not work**: measured, it reads 1,919 of 2,048
rows at small scale and 32,767 of 32,768 at flagship scale. In hundreds of
dimensions clusters are angularly so wide that `cos(phi - theta) ~ 1` for
nearly every cluster and the bound cannot discriminate. Curse of
dimensionality, and no amount of extra clusters fixed it (swept 64 → 1024).

Reported as a failure because it is one. The honest status of the head:
- exact bound pruning — **dead**, measured useless.
- approximate clustering — **promising but unproven at scale.** The 94.5%/10.3x
  figure is on a 2048-token vocabulary; my flagship-scale check used synthetic
  hidden states and gave only 70% agreement, and I do not trust that number
  either way, because real hidden states are strongly anisotropic and synthetic
  ones are not. It needs a real trained model at real vocabulary size.
- the boring option the analysis also supports — **halve the vocabulary**, a
  clean 2-4x with zero approximation, at some cost in tokens-per-character.

## Results

Two experiments on permutation composition in S₄ (each step applies a *different* permutation, so there is no wrap-around shortcut). Identical data, optimiser and seed. Chance is 4.2 %.

### Experiment 1 — trained on 1–4, tested to 9

4 M resident parameters each, 3000 steps.

| required steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 100 % | 100 % | 100 % | 93.4 % | 0.0 % | 0.0 % | 0.4 % | 2.3 % | 0.4 % |
| XIOS | 100 % | 100 % | 100 % | **100 %** | 7.4 % | 1.6 % | 4.3 % | 2.0 % | 4.3 % |

A small in-distribution win (100 % vs 93.4 % at four compositions) and **no extrapolation**: XIOS's 3.9 % extrapolation average is *below* the 4.2 % chance rate. It beats the baseline's 0.6 %, but the honest reading is "XIOS is at chance, the baseline is confidently wrong", not "XIOS extrapolates".

### Experiment 2 — trained on 1–8, tested to 9

The question Experiment 1 could not answer: is the deficit *generalisation* or *capacity*? Train on the whole range and find out. `max_iters` raised to 12, `target_depth` to 6, 2500 steps.

| required steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (6 layers, 4.4 M) | 100 % | 95.7 % | **5.5 %** | 2.7 % | 3.1 % | 5.5 % | 2.7 % | 4.7 % | 4.7 % |
| XIOS (4 stored blocks, 4.2 M) | 100 % | 100 % | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % | **48.0 %** |

**This is the depth cliff, and it is stark.** The parameter-matched dense transformer collapses to chance at *three* compositions and never recovers — six layers cannot compose more than two permutations, no matter how long it trains. XIOS, with four stored blocks looped up to twelve times, solves every trained depth perfectly and reaches 48 % on a depth it never saw (11× chance).

That is the central architectural claim — effective depth decoupled from stored depth — behaving as designed, and it is a large effect rather than a marginal one.

At matched parameters XIOS spent 2.65× the compute per token (22.50 vs 8.48 MFLOPs), which flatters it — a looped core buys compute with iterations rather than weights. So the comparison was re-run against a dense baseline given the **same FLOPs**: 16 layers, 11.8 M parameters, **2.8× XIOS's weights**.

### Experiment 3 — the same test at matched FLOPs

| required steps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (unseen) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (16 layers, 11.8 M, 22.57 MF/tok) | 100 % | 92.2 % | 22.3 % | 5.5 % | 5.9 % | 3.9 % | 2.3 % | 4.7 % | 5.1 % |
| XIOS (4 stored blocks, 4.2 M, 22.50 MF/tok) | 100 % | 100 % | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % | **48.0 %** |

**The result survives FLOP matching intact.** At 1.00× compute and with 2.8× fewer parameters, XIOS solves every trained depth while a conventional 16-layer stack collapses to chance from four compositions onward. Tripling the baseline's depth (6 → 16 layers) moved its cliff by exactly one step, from 3 to 4 — while XIOS clears all eight.

So the win is not bought with extra arithmetic, and it is not bought with parameters. It comes from *where* the depth lives: a small block stack applied many times composes operations that a larger stack applied once cannot, at the same cost.

**What this does not show.** One synthetic task, 4 M parameters, CPU. Non-parallelisable sequential composition is precisely the problem class this architecture was designed for, and therefore the one most favourable to it — natural language is mostly not that. A single clean result on a friendly benchmark is evidence, not proof, and the honest next step is the language-modelling A/B, where the effect should be much smaller or absent.

### Experiment 4 — collapsing a dense stack into a loop (the flagship path)

This is the experiment that decides whether the project has a road to a
local flagship at all, and the answer came back positive.

The architecture makes *inference* cheap but does nothing for *training* —
matching a frontier model from scratch costs millions of dollars of compute
whatever shape you choose. So the realistic route is not to train the
knowledge but to **move knowledge that already exists into this shape**. The
open question was whether a small *shared* core can represent what many
independent layers compute, since looping reuses weights and capacity is
genuinely lower.

Method (`train/collapse.py`), in two phases:

1. **Per-group regression.** A dense stack hands you a far richer signal than
   its logits: every layer's residual update. Ask iteration `i` of the loop to
   reproduce what dense layers `ik..ik+k-1` did together. That is `N` targets
   per forward pass instead of one, it is plain regression rather than language
   modelling, and it needs no original training data — only text to push
   through.
2. **End-to-end distillation.** Phase 1 feeds each group the *true* dense
   hidden state, but at run time the loop feeds itself and errors compound —
   classic exposure bias. Phase 2 runs the whole loop with no teacher forcing
   and matches the dense model's output distribution by KL.

Target: the 12-layer dense LM trained on Shakespeare (val ppl 12.83).

| | dense teacher | collapsed |
|---|---|---|
| stored blocks | 12 layers | **2 shared blocks** (6x fewer) |
| body parameters | 8.85 M | **2.66 M** (3.3x fewer) |
| val perplexity | 12.83 | **12.66** |
| perplexity retained | — | **101.4%** |

Phase 1 alone retained 88.8%; phase 2 took it to 101.4%. The collapsed model is
marginally *better* than the teacher it was distilled from — a 2-block looped
core reproduces, and slightly improves on, what 12 independent layers did.

Two things to be precise about. **Top-1 agreement is only 78.5%**, so this is
not a functional clone — it is a different model that happens to be as good.
For the purpose (keeping capability, not mimicking behaviour) that is fine, and
arguably preferable, but "101% perplexity" must not be read as "identical".
And the scale is small: a 9.4 M-parameter teacher on a 0.6 M-token corpus.
Collapsing a real 32-layer 7B model is a much larger ask. It may well go
*better* — adjacent layers in deep models are known to be more redundant than
in shallow ones, which is exactly the redundancy this exploits — but that is a
prediction, not a result.

What it does establish: **the capacity objection does not bite at this scale.**
The road from "existing open model" to "cache-resident looped core with our
bandwidth profile" is open, and it is cheap — 1,400 total steps of regression
plus distillation, no original training data required.

### Experiment 5 — collapsing a REAL pretrained model (GPT-2)

Experiment 4's 101.4% came from a teacher trained in this repo: 9.4 M
parameters on a 0.6 MB corpus. The obvious objection is that such a model is
trivially compressible. So the same procedure was pointed at GPT-2 — really
pretrained on ~40 GB of web text, and architecturally nothing like the XIOS
core (learned positional embeddings, LayerNorm, GELU MLP, versus RoPE, RMSNorm,
SwiGLU and gated linear recurrences). Corpus: *Pride and Prejudice*, 192 k
tokens, evaluated on a held-out tenth.

| teacher | compression | params | ppl retained | top-1 agree |
|---|---|---|---|---|
| toy 12-layer (repo-trained) | 12 → 2 blocks | 3.3x fewer | **101.4%** | 78.5% |
| **GPT-2, 12 layers** | 12 → 2 blocks | 5.2x fewer | **55.5%** | 52.2% |
| **GPT-2, 12 layers** | 12 → 4 blocks | 2.6x fewer | **69.4%** | 57.1% |

A third attempt (`train/absorb.py`) then changed the algorithm to attack the
diagnosis directly:

* **Scheduled self-feeding.** Phase 1 always fed the *true* dense hidden state,
  so the core never trained on the slightly-wrong inputs it actually sees.
  Feeding it its own previous output with a ramping probability, while still
  targeting the true next waypoint, asks the right question -- "given where you
  actually ended up, produce where you should be" -- and trains it to correct
  its own drift. DAgger applied to layer collapse.
* **Per-iteration low-rank adapters.** AdaLN conditioning can rescale and
  shift a shared block's behaviour but cannot change the *direction* of its
  transformation, and GPT-2's layers genuinely compute different functions. A
  rank-16 additive term per iteration costs 0.2% of the core and lets each
  iteration specialise -- the smallest unshared capacity that addresses the
  measured failure.
* Extra loss weight on the final group, whose error reaches the logits
  unattenuated.

| attempt | compression | budget (fit + distil) | ppl retained |
|---|---|---|---|
| 1 | 12 -> 2 blocks | 400 + 200 | 55.5% |
| 2 | 12 -> 4 blocks | 600 + 400 | 69.4% |
| 3 | 12 -> 4 blocks, self-feeding + adapters | 900 + 500 | **72.8%** |

The algorithmic changes bought ~3.4 points. The dominant variables were
**compression ratio and compute**, not cleverness -- worth stating plainly,
because the opposite would have been the more flattering conclusion.

**The toy result did not transfer.** On a real pretrained model the procedure
loses 30–45% of perplexity at these settings, not 0%. GPT-2's layers genuinely
do different things; my toy teacher's did not, and 101.4% was measuring that
redundancy rather than the method.

Two things the data does say, though, and both are load-bearing:

* **The compression ratio is a real, usable dial.** Going from 6x depth
  compression to 3x moved retention from 55.5% to 69.4%. There is a tradeoff
  curve here, not a cliff, and nothing forces the aggressive end of it.
* **Every run was compute-starved, and that is now the binding constraint.** Phase-2 KL was still falling
  steeply when each run ended (2-block: 1.69 → 0.93; 4-block: 0.98 → 0.53),
  with no sign of a plateau. Total budget was ~300–600 k tokens on CPU,
  against ~3 M for the toy and a model 10x larger. Distillation KL was still
  falling at the end of all three runs (0.46 and dropping on the third).
  These are **lower bounds from truncated runs**, not converged results, and
  should not be quoted as the method's ceiling.

  This matters more than it sounds: **the question cannot be settled on this
  CPU.** Three attempts, three truncated runs, loss falling in every one. The
  honest position is not "absorption retains ~73%" but "absorption retains at
  least 73% and we have never once seen it converge". The scripts take
  `--device cuda` unchanged; a free Colab T4 is roughly 30-50x faster, which
  turns 1,400 steps into 50,000 and is where this actually gets answered.

Phase 1 fidelity was never the problem: relative error on GPT-2's residual
updates reached **1.7%**. The loss appears when the loop runs on its own
output and errors compound, which is exactly what phase 2 exists to fix and
exactly what ran out of compute.

### Experiment 6 — what redundancy actually means, and what absorption is worth

Three more attempts and one diagnostic changed the picture substantially. The
diagnostic (`train/redundancy.py`) should have come first; running it after
three blind attempts is the process error worth recording.

**Measured on GPT-2, no training required:**

| probe | result |
|---|---|
| weight-space cosine between layers | **0.009** — essentially orthogonal |
| layer pairs that substitute better than doing nothing | **13 / 132** |
| cost of dropping any single layer 1-10 | **1.08-1.33x** perplexity |
| cost of dropping layer 0 | **134x** |
| cost of dropping layer 11 | **3.4x** |

This resolves the confusion in one stroke: **a model's redundancy is
"expendability", not "interchangeability."** Any single middle layer can be
deleted for almost nothing, yet almost no layer can stand in for another. Those
are completely different properties, and the whole approach had been assuming
the second while the evidence only supports the first.

Three consequences, each confirmed by a run:

* **Averaging layer weights to warm-start a shared block is meaningless.**
  At cosine 0.009 it produced a perplexity of **351,502** and 0% agreement.
  Warm-starting from a *single* representative layer instead gave a usable
  starting point (perplexity 426, then 55 in the 2-iteration configuration).
* **Boundary layers must be kept verbatim.** Layer 0 at 134x cannot be covered
  by the same shared block as the interchangeable middle.
* **The governing variable is the number of shared iterations, not the
  parameter ratio** — because error compounds each time the core feeds itself:

| shared iterations | param ratio | ppl retained |
|---|---|---|
| 6 | 5.2x | 55.5% |
| 5 | 3.0x | 53.4% |
| 3 | 2.6x | 72.8% |
| 2 | 1.7x | 74.3% |

Retention sits near 54% at 5-6 iterations and near 73% at 2-3, and relaxing
compression further does not break a ceiling around **74%** on this hardware.
Unlike the earlier runs, distillation KL had flattened (~0.5-0.6), so this is
closer to converged than compute-starved.

### Is 74% good or bad? The baseline that settles it

"74% of GPT-2's perplexity" sounds like a failure in isolation. It needs the
obvious alternative for comparison: just **delete** the droppable layers.
Measured, with no retraining at all:

| depth compression | pruning (free) | absorption (trained) |
|---|---|---|
| 1.20x | 87.9% | — |
| 1.33x | 78.8% | — |
| 1.50x | 64.4% | — |
| 1.71x | **44.2%** | **74.3%** |
| 2.00x | **28.8%** | — |
| 2.60x | — | **72.8%** |

At 1.71x, absorption beats pruning **74.3% vs 44.2%**. At 2.0x compression
pruning has collapsed to 28.8%, while absorption at *more* compression (2.6x)
holds 72.8%. **Absorption is roughly 2.5x better than the obvious baseline at
comparable compression** — so it is doing real work, not merely losing
gracefully. Equally: if you only want ~1.2-1.3x, pruning is free and better,
and there is no honest reason to run absorption at all.

### Where this leaves absorption

The claim that survives is narrower and more useful than the one I started with:

* **Loop-from-scratch works well.** A model *trained* in this shape solves
  3-to-8-step composition at 100% where dense transformers sit at chance, at
  matched FLOPs and 0.35x the parameters (Experiments 2-3).
* **Loop-by-absorption is lossy but competitive.** Moving an existing
  pretrained model into the shape costs ~25% of its perplexity at 2-2.6x depth
  compression, which is far better than pruning but far from free. It is a
  real tool, not a free lunch.

Those are different use cases, and conflating them is what made the toy's
101.4% look like a general result.

### The structural limit this exposed: depth compresses, width does not

Working through what absorbing a *large* model would actually require surfaced
a limitation more fundamental than the compute budget, and it reshapes the
roadmap.

**Collapse compresses depth. It does nothing about width.** A 70 B model is
large mostly because it is *wide* — ~8192 hidden dimensions with a ~22 k FFN —
not because it has 80 layers. Collapsing 80 layers into 8 shared blocks is a
10x cut in stored blocks, but each block is still 8192-wide, so the result is
still billions of parameters. Useful (≈3.6 GB at int4) but **not
cache-resident**, and cache-residency is the entire source of the 55x
bandwidth advantage. The two wins pull against each other.

So a genuinely flagship-class local model needs width reduction too, and that
is where the disk-resident memory stops being an optional third mechanism and
becomes the crux. Most of a wide FFN is *knowledge storage*, and knowledge is
precisely what belongs on an SSD rather than in RAM. The full pipeline:

1. **Collapse depth** into a looped core — partially proven (55–69% on a real
   model, under-trained, ratio-tunable).
2. **Move FFN knowledge to the disk store** — the untested crux. This is what
   would shrink *width* without discarding what the model knows.
3. **Quantise the remainder** to int4 — proven, 3.3-3.4x.
4. Result: narrow cache-resident reasoning core + large disk knowledge store.

Steps 1, 3 and 4 have measurements. Step 2 has none, and it is now clearly the
single highest-value experiment in the project — not because it is elegant, but
because without it the bandwidth advantage and the absorption path cannot be
had at the same time.

### Retracted: the adaptive-allocation result

An earlier version of this README reported a **+1.000 correlation between problem difficulty and allocated depth** as the headline finding — the controller discovering, unprompted, that harder problems deserve more thinking.

**That was a measurement artefact and it is withdrawn.** `exact_match` averaged depth over *all* positions, including padding, and padding fraction varies with problem size — so the statistic was largely measuring prompt length. Re-measured on the tokens that actually carry the computation:

| | reported (all tokens) | corrected (scored tokens) |
|---|---|---|
| Experiment 1 | +1.000 | **−0.256** |
| Experiment 2 | −1.000 | **+0.411**, depth flat at 6.00 |

Depth sits at the budget target (2.94–2.98 and 6.00 respectively) essentially independent of difficulty. **The allocation is not difficulty-adaptive.** Two contradictory "perfect" correlations from the same mechanism should have been the tell; the sign flipped with padding, which no real effect would do.

So what actually produced Experiment 2's win is **weight-shared looped depth**, not adaptivity. The looping is doing the work; the adaptive part is, so far, decorative. `PonderStats` now separates `mean_depth`, `scored_depth` and `compute_depth`, and `exact_match` reports both figures, so this particular way of fooling oneself is closed off.

### Where that leaves the three mechanisms

| mechanism | status |
|---|---|
| 1a. Weight-shared looped depth | **Works, large effect, confirmed at matched FLOPs.** 100 % vs 2–6 % past the dense cliff, at 1.00× compute and 0.35× the parameters. |
| 1b. Budget-allocated *adaptive* depth | **Not supported.** Depth is flat at the budget; allocation is not difficulty-sensitive. |
| 2. Per-depth memory timelines | Implemented and verified consistent (train ≡ decode to 1e-6). No ablation yet, so its contribution is unmeasured. |
| 3. Disk-resident knowledge | Mechanically verified (int8 mmap ≡ RAM to 0.6 %, 1.83× fewer page reads). Never tested for whether it helps quality. |

Next, in order of information per GPU-hour: the language-modelling A/B, where this effect should mostly vanish and the result will be far less flattering; an ablation of the looped core against a plain unrolled stack with no halting, to isolate whether *any* of the ponder machinery earns its place given that adaptivity is currently doing nothing; and an ablation of per-depth timelines against a single shared timeline.

## The falsification test

```bash
python train/ab_experiment.py --task perm --preset nano \
    --steps 4000 --train-max-steps 6 --eval-max-steps 14
```

Two models, **identical resident parameter count**, identical data, optimiser and seed. One is a standard transformer with all its depth baked in — the harness gives it the *deepest* stack the matched budget buys, since depth is precisely what XIOS claims to get for free. The other is XIOS. The disk memory is disabled for this comparison, so it isolates the depth mechanism rather than handing XIOS a knowledge store the baseline has no equivalent of.

Both train on problems needing 1–6 sequential steps; both are tested to 14. The tasks (`xios/tasks.py`) are generated, so no contamination, and each has a tunable number of required steps: iterated permutation composition in S₅ (non-abelian, so the chain provably cannot be parallelised), multi-hop pointer chasing, modular arithmetic chains, and gated parity — the standard separation example for bounded-depth circuits.

**Confirms the hypothesis:** XIOS degrades gracefully past the training range where the baseline falls off a depth cliff, *and* measured mean depth rises with difficulty. Nothing in the loss asks for the second one; if that curve slopes up, the controller found it alone.

**Kills it:** comparable or worse extrapolation, or a flat depth-vs-difficulty curve (adaptivity is decorative), or a win that evaporates once compute is equalised instead of parameters. `xios/flops.py` reports both, because equal-parameter is the comparison XIOS wants and equal-FLOP is the one that could sink it.

A nano run is ~20 minutes on a free Colab T4. If it loses there, it's wrong, and we know in an afternoon instead of a quarter.

---

## Trying it

```bash
python app.py --ckpt runs/demo/xios.pt --tokenizer runs/demo/tokenizer.json
```

A Tkinter UI (stdlib only). Two things in it are worth more than the chat box:

**Thinking is literal.** XIOS decides per token how many times to loop its
core, so "thinking" is an integer the model emits, not a hidden scratchpad.
Each generated token is shaded by the depth it cost and the right-hand panel
reports the live distribution. Tick **web search** to retrieve DuckDuckGo
results, which are packed into a context preamble and shown verbatim, so you
can see what the model was actually given.

To train the small demo model used above (~90 min on CPU, minutes on a GPU):

```bash
python train/train_text.py --dataset shakespeare --preset nano     --dim 256 --core-blocks 2 --n-kv-heads 2 --vocab 2048     --steps 1600 --batch-size 24 --seq-len 128 --max-iters 6     --target-depth 3 --only xios --out runs/demo
```

4.7 M parameters, val perplexity 13.3, mean depth 3.01 of a 6-iteration ceiling.

### Where a trained model spends its thinking

`python train/depth_profile.py` groups generated tokens by the compute they
cost. On the demo model this is clean and interpretable — and unlike the
synthetic-task measurement, these are *generated* tokens each carrying their
own depth, so no padding or sequence geometry can fake it:

| token kind | mean depth (ceiling 6) |
|---|---|
| whitespace | 2.80 |
| word-internal | 3.80 |
| punctuation / structure | **6.00** |

```
depth 2 (54%)  ' ' ' ' ' ' 'T' ' ' ...          cheapest: spaces
depth 3 (13%)  'to' 'know' 'master' 'do' 'not'
depth 5 ( 8%)  'the' 'VINCENTIO:' 'thy' 'Why,'
depth 6 (20%)  '?' '
' '.' 'A:' 'my'           ceiling: line breaks, turns
```

Every punctuation or structural token runs to the ceiling. In a play script
those *are* the hard calls — when does the line end, who speaks next — while a
space is free. This is a different and weaker claim than the one retracted
below (it is about token type, not task difficulty), but it is measured
cleanly and it is the first sign the allocation does something sensible on
natural text.

## Quick start

```bash
pip install -r requirements.txt

python -m xios.cli info                  # sizes, depth leverage, disk footprint
python -m xios.cli bench --preset nano    # decode speed + depth histogram
python -m xios.cli quantize --preset small

python tests/test_recurrence.py           # the invariants
python tests/test_decode_consistency.py
python tests/test_ponder.py
python tests/test_memory_locality.py
python tests/test_quant.py
python tests/test_memory_disk.py
python tests/test_learns.py       # the floor: XIOS must match a dense baseline

python app.py --selftest          # UI generation + search path, no window
python -m xios.search "query"     # search alone
```

`notebooks/XIOS_Colab.ipynb` runs the whole programme on a free T4.

## Layout

```
xios/
  config.py          presets; effective-depth / stored-depth ratio
  model.py           XiosChat: prelude → core → coda, prefill/decode runtime
  baseline.py        matched conventional transformer (the opponent)
  tasks.py           depth-sensitive task families
  flops.py           compute accounting for equal-FLOP comparison
  tokenizer.py       byte-level BPE, no dependencies
  cli.py             info / bench / quantize / chat
  core/
    recurrent.py     gated linear recurrence: chunk-parallel + true recurrent
    attention.py     sliding-window GQA over real token positions
    ponder.py        halting, two-sided budget loss, budget curriculum
    rcc.py           the looped core + per-depth memory timelines
    block.py  norm.py
  memory/dam.py      product-key memory, mmap int8, seriated Z-order layout
  quant/             int4 groups, searched AWQ, int8 embeddings, tied head
train/
  train.py           shared loop, so neither model gets an optimiser edge
  ab_experiment.py   the falsification test
  train_text.py      language-modelling A/B
app.py               desktop UI: live ponder depth + web search
xios/bandwidth.py    bytes/token accounting + config designer (start here)
xios/head.py         clustered output projection (the head bottleneck)
xios/search.py       dependency-free DuckDuckGo scrape
train/collapse.py    dense stack -> looped core (the flagship path)
train/collapse_hf.py collapse a REAL pretrained model (GPT-2, Llama, ...)
train/absorb.py      absorption: self-feeding + per-iteration adapters
train/absorb2.py     absorption with the teacher's own block type, boundaries kept
train/redundancy.py  measure layer redundancy FIRST (run this before absorbing)
train/memory_quality.py  does the disk store buy quality? (with frozen ablation)
train/fact_capacity.py   arbitrary-fact recall per resident byte
xios/adapters.py     wrap any HuggingFace causal LM for collapse
train/depth_profile.py  where a trained model spends its thinking
train/summarize.py   compare A/B runs, chance-relative
tests/               every claim in the tables above
  test_recurrence.py         parallel form == sequential form
  test_decode_consistency.py training semantics == generation semantics
  test_ponder.py             causality, budget, real sparsification
  test_memory_locality.py    page reads, bijection, layout cost
  test_memory_disk.py        export -> mmap -> same answers
  test_quant.py              size claims, AWQ search, tied head
  test_learns.py             the floor: XIOS matches a dense baseline
  test_bandwidth.py          bytes/token, the cache rule, the `base` trap
```

### Experiment 7 — the disk-resident memory does not work (as implemented)

This was the crux. The bandwidth result forces the looped core to stay inside
L2/L3, which caps its *width*; knowledge needs bits; so the product-key store
on disk was the only place knowledge could go. Everything about it up to this
point was *mechanical* — int8 export round-trips to 0.6%, addressing is a
verified bijection, page reads drop 1.83x — and none of that is evidence that a
model can learn to **use** it.

Two independent probes, each with the ablation that makes it a real experiment.

**Probe 1: language modelling** (`train/memory_quality.py`). Same data, seed and
budget; core FFNs replaced by the store.

| variant | val ppl | resident | disk |
|---|---|---|---|
| dense FFN | **14.43** | 2.6 MB | — |
| memory (learned values) | 14.79 | 2.4 MB | 4.2 MB |
| memory (values **frozen at random**) | 14.91 | 2.4 MB | 4.2 MB |

Slot utilisation was excellent — **100% and 99.8% of slots touched**, usage
entropy 0.70-0.73 against 1.0 for uniform — so load balancing works and
capacity is not collapsing onto a few slots. But the decisive number is the
ablation: **learned values beat frozen random values by 1.008x.** The contents
of the store are irrelevant. It is functioning as a random-feature projection,
not as a memory.

**Probe 2: arbitrary fact recall** (`train/fact_capacity.py`). Shakespeare
contains almost no rare facts, so probe 1 arguably could not detect knowledge
storage at all. So: generate `N` key->value pairs with no structure whatsoever,
where memorisation is the only possible strategy, and measure recall.

| facts | dense | memory | frozen |
|---|---|---|---|
| 1,000 | 100% | 100% | 100% |
| 5,000 | 0.6% | **0.2%** | — |

No advantage at either point. And note what the cliff from 100% to 0.6% means:
5,000 facts is 6.2 KB of information against 1.28M parameters, so this is not a
capacity limit. It is an **exposure** limit — at a fixed 2,000 steps each fact
is seen ~128 times at 1k facts but only ~26 times at 5k. So this probe measures
memorisation *per training budget*, not raw capacity, and the 20,000-fact row
was abandoned because both arms were already at the floor.

**Second attempt: fix the addressing.** `train/memory_diagnose.py` found two
real mechanical faults, and neither was the obvious one (gradient starvation —
the value gradient was 2.2e-1, not zero):

* **Averaging washout.** Retrieval-softmax entropy was **3.326 of a maximum
  3.466** — 96% of uniform — so the output was ~the mean of 32 value vectors
  and carried almost no information about *which* slots were chosen. Cause:
  queries were unit-normalised while keys were initialised at std 0.02, giving
  score differences of ~0.16.
* **Content-independent routing.** Different tokens retrieved **69.6%** of the
  same slots. LayerNorm normalises each query individually and does nothing to
  decorrelate queries *across* tokens, so one shared direction sent everything
  to the same place. Addressing cannot be associative if every input addresses
  the same slots.

Both were fixed — keys at `key_dim**-0.5`, a learnable softmax temperature, and
**BatchNorm** over the query features (what the product-key literature uses).
The fixes demonstrably worked:

| metric | before | after |
|---|---|---|
| retrieval-weight entropy (max 3.466) | 3.326 | **1.185-1.622** |
| mean top weight | 0.091 | **0.549-0.675** |
| slot overlap between tokens | 69.6% | **32.7-45.6%** |
| value gradient norm after 300 steps | 1.0e-3 | **3.6e-3** |

**And quality did not move.** Perplexity 14.79 -> 14.98, and learned values
versus frozen random values went from 1.008x to **0.992x** — now marginally
*worse* than random. Fixing the addressing changed the mechanism's internals
completely and changed its usefulness not at all.

That is a much stronger result than the first negative, because it eliminates
the obvious explanation: the addressing was never the bottleneck. At this scale
there is simply nothing for a memory to store — the dense path already captures
everything 0.6M tokens of Shakespeare contains, and a knowledge store only pays
when there are rare facts that cannot be compressed into weights. The
fact-recall probe agrees: no advantage at 5,000 facts either.

**A lesson worth keeping:** every *mechanical* health metric can look fine, or
be repaired, while the component still does nothing. I read "100% of slots
touched, entropy 0.73" as success when it was meaningless, then fixed two real
bugs that turned out not to matter. Only the end-to-end ablation — replace the
learned thing with a random thing and see if anything changes — was ever
decisive. Build that ablation first, and trust nothing else.

**Conclusion: mechanism 3 is not earned, and the frozen ablation is why.** The
scale caveats are real — 1.3-4.7M parameters, 0.6M tokens, and product-key
memories are reported to need large models to pay off — but a scale argument
cannot explain a 1.008x gap against *random frozen values*. At this scale the
store's contents carry no information, full stop.

**What this does to the design.** The width problem is now unsolved, and with it
the knowledge story for the `flagship` preset: a narrow cache-resident core with
nowhere to put facts. Three options, in the order I would try them:

1. **Retrieval over documents instead of learned slots.** Strategically this was
   already the better answer (see the roadmap below) and it is now empirically
   motivated: knowledge that lives in *text on disk* needs no learned addressing
   at all, and the retrieval machinery in `xios/search.py` already works. This
   is the honest pivot.
2. **Re-test the store at scale on a GPU**, where product-key memory is reported
   to work. Cheap to try once the GPU path exists, and it would be wrong to
   declare the mechanism dead on 4M-parameter evidence alone.
3. **Cut it.** If neither of the above pans out, the memory is 800 lines that
   earn nothing and should be deleted rather than defended.

### Experiment 8 — the adaptive-depth controller is harmful; cut it

The roadmap said a mechanism that does nothing should be made to work or
deleted, not defended. So: same config, same seed, same data, only the
controller differs. `adaptive_depth=False` runs **exactly** `target_depth`
iterations for every token — no halting head, no budget loss, no Lagrangian,
no annealing schedule.

| required steps | adaptive depth | **fixed depth** |
|---|---|---|
| 1-8 (trained) | 100% | 100% |
| **9 (unseen)** | **48.0%** | **99.6%** |

Compute is identical (22.50 vs 22.51 MFLOPs/token) and fixed depth converges
*faster* (training loss reaches 0 by step 1750 against ~2000).

**The controller was not decoration, it was a liability.** Deleting it roughly
doubles extrapolation accuracy. The mechanism is plausible in hindsight: a
halting controller trained on 1-8-step problems learns to stop at its budget,
and then *caps* the compute available to a 9-step problem — precisely the case
where more iterations were needed. Fixed depth guarantees every token gets the
full allowance. Adaptive allocation only helps if the allocation is right, and
here it was systematically wrong in the direction that matters.

This retires a large amount of machinery: the halting head, the two-sided
budget loss, the Lagrange multiplier and its dual-ascent tuning, the budget
annealing curriculum, the soft/hard halting reconciliation, and the per-depth
`eps` cutoff. All of it existed to solve a problem — allocating compute by
difficulty — that never materialised and cost accuracy to attempt.

What survives is the part that was always doing the work: **a weight-shared
core, looped a fixed number of times, sized to fit in cache.** That is a
simpler architecture than the one this project started with, and a better one.

`adaptive_depth` remains a config flag rather than a deletion, because the
mechanism might earn its place at scale or with a better-designed controller
(the soft mixture was already replaced once, by update-gating, for a similar
reason). It now defaults to what the measurement supports.

## Roadmap: what would actually close the gap

Written after the absorption results, because they changed what the right
target is.

### The strategic reframe

The arithmetic on "absorb a frontier model" does not work. Absorption costs
~25% of perplexity, and you can only absorb what is openly available, so the
best case is `(best open weights) x (absorption loss)` — which is not frontier.
Chasing a frontier model's *knowledge* with this architecture is the wrong race,
and no amount of engineering fixes it: knowledge is bits, frontier models have
an enormous number of them, and training those bits costs ~10^25 FLOPs that
this design does nothing to reduce.

But knowledge is also the part that does **not** have to live in weights. For a
local model, retrieval over a local corpus and the web is not a workaround, it
is the correct design. A small reasoner with good retrieval beats a large model
at recall while being orders of magnitude cheaper.

What this architecture is unusually good at is the *other* axis: **sequential
compute per byte read**. Measured, a cache-resident looped core reads 55x fewer
bytes per token than a 3B dense model while applying ~200 block-applications.
And the frontier's recent reasoning gains come largely from *test-time*
compute — long chains of thought, search, verification — which is precisely the
resource this design makes cheap.

So the goal worth pursuing is not "know what Opus knows". It is: **out-think a
frontier model per dollar of local hardware, and retrieve what you do not
know.** That is a defensible target, and it plays to a structural advantage
rather than against a structural disadvantage.

### Priority order (information per unit of effort)

1. ~~**Test the disk-resident memory for quality.**~~ **Done — it failed.**
   See Experiment 7: learned values beat frozen random values by 1.008x, and it
   gave no advantage on arbitrary fact recall. The width problem is therefore
   open, and the knowledge story has moved to retrieval over documents (which
   was already the better strategic answer). Re-test the store at scale on a GPU
   before deleting it, but do not build on it.

2. **Get off the CPU.** Every result here is at =<5M parameters, and several
   runs ended with loss still falling. Nothing about language quality has been
   tested at a scale where it means anything. This is logistics, not research:
   the scripts take `--device cuda` unchanged, and a free T4 is 30-50x faster.

3. **Test looping on real reasoning, not synthetic composition.** The 100%-vs-
   chance result is on permutation composition, the friendliest possible task
   for the mechanism. GSM8K-style multi-step problems are the claim that
   matters. This could deflate the headline result, which is exactly why it is
   worth running early.

4. ~~**Ablate the ponder controller.**~~ **Done — cut it.** See Experiment 8:
   fixed depth beats adaptive 99.6% vs 48.0% on unseen problem depth, at equal
   compute, and converges faster. `adaptive_depth=False` is the supported
   setting now. Revisit only if a better controller design is proposed.

5. **Then, and only then, long-chain-of-thought training on the looped core.**
   Where the frontier's reasoning gains actually live, and where cheap
   test-time compute compounds. Pointless before 2 and 3.

### What would make this genuinely change things

If a ~500M-parameter cache-resident core, with knowledge on disk and good
retrieval, reasons as well as a 30-70B dense model at 30x the local speed —
that is a result that matters, and every component of it now has either a
measurement or a named experiment. It is not Opus. It might be the best thing
that runs well on a laptop, which is a different and more achievable prize.

## Status

Runs end to end on CPU. All invariant tests pass, including the floor test where XIOS matches a dense baseline exactly.

**What is earned:** weight-shared looped depth is a large, real effect, and it survives the comparison designed to kill it. A 4.2 M-parameter XIOS solves 3-to-8-step permutation composition at 100 % where dense transformers sit at chance — both a parameter-matched 6-layer model *and*, at identical FLOPs, a 16-layer model with 2.8× the weights. Tripling the dense model's depth moved its cliff by one composition; XIOS clears all eight, and reaches 48 % at a depth it never trained on. That is the architecture's central claim working, measured the hard way.

**Partially earned:** absorbing an existing model into the shape. It costs
~25% of GPT-2's perplexity at 2-2.6x depth compression — not free, and nothing
like the 101% a toy teacher suggested. But against the obvious alternative it
is ~2.5x better (72.8% at 2.6x compression, versus pruning's 28.8% at 2.0x),
so it is a genuine tool. The governing variable turned out to be the number of
shared iterations, not the parameter ratio, because error compounds each time
the core feeds itself.

**Not earned, and now measured as such:** the disk-resident memory. Its
contents are irrelevant — learned values beat frozen random ones by 1.008x — so
at this scale it is a random-feature layer wearing a memory's clothes. That
leaves the width problem open and the `flagship` preset without a knowledge
story, which is the most consequential gap in the project.

**Disproven, and removed:** the adaptive part of adaptive depth. It was ablated, and fixed depth won decisively — **99.6% versus 48.0%** on unseen problem depth at equal compute, converging faster. The halting controller was not neutral; it capped compute exactly where more was needed. The looping was always doing the work. Also unearned: anything about natural language, where this effect should be far smaller, and anything at a scale beyond 4 M parameters.

**What the project is, honestly:** a novel architecture whose looping mechanism demonstrably buys depth that parameters alone do not, wrapped in an apparatus that has now caught eleven of my own errors — including two bugs in the central mechanism, a benchmark with a shortcut, a budget that priced the wrong quantity, and a headline result that was noise. The guards exist because each one caught something. That is what makes the next result worth believing.
