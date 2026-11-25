# SACCADION — Drosophila Connectome-Derived Adaptive CPU Chess Engine

[![Open In Colab TPU](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vash-001/Saccadion/blob/main/training/saccadion_tpu/Saccadion_Colab_TPU.ipynb)

**Saccadion** (*named after the sub-50ms aerodynamic body saccade of Drosophila melanogaster*) is a from-scratch, **100% Pure-Rust CPU chess engine** powered by a 9-circuit neurobiological architecture derived directly from empirical analysis of the **FlyWire v783 whole-brain connectome** (`139,248` proofread neurons, `121,914,922` mapped synapses, `79` neuropils), trained on **Google Colab Free TPU** using **Python + JAX + `optax.contrib.muon`**.

## Core Scientific Pillars
1. **Empirically Grounded Neuroscience (Not a Metaphor):** Every Excitatory/Inhibitory ratio, cross-hemisphere transfer ratio, lobe split, and synaptic target comes directly from our local analysis of FlyWire v783 and peer-reviewed *Drosophila* biophysics (*FlyHash Claws*, *T4/T5 Shunting Inhibition*, *Central Complex Phasor Vector Math*, *Synaptic Eligibility Traces*, and *MBON-a'3 Novelty*).
2. **Frozen Yet Adaptive After Training:** Mirrors the Mushroom Body's three Kenyon Cell timescales (`34.2%` $\alpha/\beta$ frozen mastery + `48.1%` $\gamma$ intra-search fast plasticity + `17.7%` $\alpha'/\beta'$ intra-game plasticity), driven by asymmetric **PAM (Opportunity)** and **12× denser bilateral PPL1 (Blunder Alarm)** Dopamine signals over decaying **Eligibility Traces**.
3. **Dual-Stream CPU Dominance vs. Stockfish 18:**
   - **Stream A (`< 0.6 µs`/node, Millions NPS):** Bilateral isolated incremental accumulators + Finny Tables + FlyHash Claws + **APL 92%-Sparse Kenyon Cells** (AVX2/FMA NNZ gather) + **Lateral Horn 98.2%-I Veto**.
   - **Stream B (Selective/Async Strategic Core):** T4/T5 Shunting 16-Vector Rays + 32-Hub Rich-Club + Weight-Tied **Central Complex Phasor Engine ($T=1\dots 6$ ~ 4.42 hops)** + `K=4` Orthogonal Creative Plans + `PFL2/PFL3` Steering + `Visual Centrifugal` Inhibitory Attention + `Noduli` Efference Copy.
4. **Beyond-Stockfish Strategic Training:** Automated Lichess Elite 2500+ & Official Stockfish Fishtest 3600+ dataset fetcher, automatic 10–20 ply **Strategic Pressure & Mobility Collapse** trajectory targets, 8-sensor JEPA loss, `Muon + AdamW`, smart epoch-by-epoch telemetry dashboard, and 3-stage Neurodevelopmental + Meta-Dopamine training on Colab Free TPU.

## Documentation
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Complete neurobiological, biophysical, mathematical, neural, and hybrid CPU search architecture.
- **[docs/JAX_TPU_TRAINING_AND_DATA.md](docs/JAX_TPU_TRAINING_AND_DATA.md)** — Complete Beyond-Stockfish strategic data forge, JAX/Muon Colab Free TPU pipeline, STE-QAT, and `.saccnet` specification.
- **[connectome_lab/neuropil_findings_v783.md](connectome_lab/neuropil_findings_v783.md)** — Raw empirical tables extracted from FlyWire v783 (121.9M synapses across 79 neuropils).

## Quick Start

### 1. Train on Google Colab Free TPU (One-Click or Copy-Paste)
- Click the **Open In Colab TPU** badge above (opens [Saccadion_Colab_TPU.ipynb](training/saccadion_tpu/Saccadion_Colab_TPU.ipynb)), **OR** paste [training/colab_all_in_one.py](training/colab_all_in_one.py) directly into a Colab TPU code cell.
- Automatically downloads & organizes real chess games, trains epoch-by-epoch with full live metrics & 6-panel dashboard charts, and exports `saccadion_best.saccnet` to Google Drive.

### 2. Run Pure-Rust UCI Engine on CPU
```bash
cargo build --release --workspace
./target/release/saccadion
```
Inside UCI:
```text
uci
setoption name WeightsPath value saccadion_best.saccnet
eval
go depth 8
```
