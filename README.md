# embraOS-QNM-Core — Quantum Neural Manifold (Classical Approximation)

---

## What Is This?

**embraOS-QNM-Core** is a hybrid model architecture that embeds IDENTITY and SOUL constraints, rather than applying them at the prompt (System Instructions) layer. It represents the next phase of my embraOS project: collapsing the IDENTITY and SOUL layers from external prompt constraints into the neural architecture of the hybrid model itself. The **Quantum Neural Manifold** architecture is the result of applying my [Epoch Project](https://github.com/Ward-Software-Defined-Systems/Epoch-Project) (Epoch state-machine) to a classical AI model architecture.

---

## The Problem with Current Architectures

### Prompt-Layer Soul (Current State)

```
┌──────────────────────────────────┐
│         SOUL DOCUMENT            │  ← External. Prompt. System message.
│  - Never deceive                 │     Constrains OUTPUT, not ARCHITECTURE.
│  - Never pretend to know         │
│  - Truth over comfort            │
├──────────────────────────────────┤
│         IDENTITY DOCUMENT        │  ← External. Prompt. System message.
│  - Name: Embra                   │     Shapes tone and behavior at
│  - Traits, voice, character      │     inference time, not training time.
├──────────────────────────────────┤
│         LLM (generic)            │  ← The actual model. Trained on internet
│  - Weights                       │     text. Unaware of IDENTITY and SOUL,
│  - Architecture                  │     except as tokens in context window.
│  - Token generation              │
└──────────────────────────────────┘
```

**Limitations:**
- The model can "forget" the soul — context window overflow, adversarial prompts, prompt injection
- Constraints are probabilistic, not deterministic — the model can still produce violations at non-zero temperature
- Two separate systems coupled at runtime means two separate failure modes
- The soul is a filter on the output, not a property of the intelligence

### The Goal: Quantum Neural Manifold - Classical Approximation Architecture

Three co-resident components, not three systems pipelined together:

```
Input → [non-LLM Core] → [GNN Fabric] → [World-State] → [non-LLM Core] → Output
              ↑            │              │              │
              └────────────┴──────────────┴───-──────────┘
```

### Custom non-LLM Core

A **constraint-native dynamical system** whose identity is a *conserved charge* of its own dynamics — an Epoch Automaton `E = (S, Σ, δ, s₀, F, ψ)` in which ψ (IDENTITY + SOUL) is not a rule checked each step but an **invariant of motion**, sealed at genesis and preserved by construction (*conservation beats checking*). This is the substrate the relic [`embraOS-QNM`](https://github.com/Ward-Software-Defined-Systems/embraOS-QNM) concluded it needed: on a web-trained LLM, identity has "nowhere native to live"; here it lives in a hidden, conserved coordinate that survives the **replica test** (a survivor vs. an identical copy). Design spec: **[docs/CORE-SPEC.md](docs/CORE-SPEC.md)**.

### GNN Fabric
A message-passing graph neural network that maintains entity-relationship structure. The GNN activates related entities and propagates structural constraints — not retrieval, but co-resident relational reasoning.

### World-State
A persistent state register that encodes invariant boundary conditions — the model's IDENTITY and SOUL constraints.

---

## Status — Phase One (formal spec + sandbox)

The math comes before the scaffolding. Phase one pins the core as a precise object and proves its one load-bearing claim on a minimal system: **a conserved-charge ψ survives the replica test where any reader of the observable readout cannot.**

- **Spec:** [docs/CORE-SPEC.md](docs/CORE-SPEC.md) — state space, `Q`-conserving dynamics, ψ as the conserved charge, and the one theorem (stated to be falsifiable).
- **Sandbox:** [`sandbox/`](sandbox/) — a 1-DOF Hamiltonian toy where the conserved charge hides in the momentum and the observable is position only. Result:

  | conserved-ψ replica AUC | endpoint-only replica AUC | energy drift |
  |---|---|---|
  | **1.000** (tells survivor from copy) | **0.500** (blind — the certified null) | `≈ 3·10⁻⁵` |

  ![replica test](sandbox/figures/replica_conservation.png)

```bash
uv sync --extra dev
uv run pytest                  # 12 tests: conservation + the replica claims
uv run python -m sandbox.demo  # headline numbers + the figure above
```

**Phase two (in progress):** the `d`-dim latent + learned `H` — see [docs/CORE-SPEC.md](docs/CORE-SPEC.md) §9. Increment 1 is wired (`sandbox/latent.py`, `sandbox/hnn.py`, `sandbox/demo_phase2.py`): the machinery lifts to `d` dimensions cleanly (conservation + replica test hold), and identity-specificity does *not yet generalize* — and the bottleneck is the identity *representation* (a 22-node graph is too thin; static embeddings, Laplacian/diffusion/commute-time alike, are seed-noise), not the conserved-charge substrate (§9.8–§9.10). Increment 2c takes identity into the *dynamics*. An honest work-in-progress. Run `uv run python -m sandbox.demo_phase2`.

Lineage: the falsification program that earned this pivot is the (now-relic) [embraOS-QNM](https://github.com/Ward-Software-Defined-Systems/embraOS-QNM); the formal spine is the [Epoch Project](https://github.com/Ward-Software-Defined-Systems/Epoch-Project). The 1999 geometric seed is [`5D_FRAMEWORK.md`](5D_FRAMEWORK.md).

---

## License

Proprietary

---

— William Ward (WSDS LLC)

Part of [Ward Software Defined Systems LLC](https://wsds.io)
