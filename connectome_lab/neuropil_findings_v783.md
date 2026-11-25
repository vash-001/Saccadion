# FlyWire v783 — 79-Neuropil Synaptic Projection Analysis (121.9M Synapses)

> **Source Data:** `connectome_lab/data/neuropil_pre_783.feather` joined with `flywire_neurons.tsv` via `connectome_lab/analyze_neuropils.py`.
> **Coverage:** `138,644` proofread neurons, `79` neuropil compartments, **`121,914,922` mapped pre-synaptic outputs**.

## 1. Key Empirical Laws Extracted from Real Synaptic Weights

| Biological Finding | Exact Measured Value | Architectural Law for Our Chess Engine |
| :--- | :--- | :--- |
| **True Synaptic Budget** | `central` **47.79%** (`58.3M` syn, `1,800`/neu) vs `optic` **35.85%** (`43.7M` syn, `564`/neu) | Dense reasoning capacity belongs in the Central Strategic Core; visual layers stay thin and weight-shared. |
| **Strict Hemispheric Isolation at Entry** | `optic` **97.5% Ipsi**, `T4/T5` **100% Ipsi**, `KC` **99.8% Ipsi**, `LHLN` **100% Ipsi** | Side-to-Move and Opponent perspectives process eyes, sparse Kenyon memory, and innate Lateral Horn vetoes **independently** (zero premature mixing!). |
| **Strategic Cross-Talk at Readout (`MBON`)** | `96` MBONs, **`6,088` syn/neu**, **66.6% Ipsi / 32.6% Contra**, targeting `CRE 26.5%`, `SMP 24.8%`, `SIP 9.9%`, `LAL 7.3%` | My valuation and opponent's valuation merge at a **2:1 push-pull cross-talk ratio** inside the Convergence Hub (`SMP/CRE/SIP`). |
| **PPL1 Blunder Alarm = 12× Denser than PAM** | `DAN_PPL1` (`16` neu): **`3,182.5` syn/neu** (`33.2%` Contra) vs `DAN_PAM` (`307` neu): `263.7` syn/neu (`65.4%` → `MB_ML`) | Wide fine-grained positive strategic rewards (PAM) + ultra-high-gain bilateral tactical blunder alarm (PPL1) that warns **both** perspectives immediately. |
| **Giant Single-Neuron APL Gate** | `2` APL neurons: **`96,894` syn EACH** (`99.9%` Ipsi, 100% GABA) | One global lateral-inhibition normalizer per hemisphere enforcing ~92% exact zeros in Kenyon Cells. |
| **Innate Lateral Horn Veto (`LHLN` / `LHCENT`)** | `LHLN` (`514` neu, **98.2% I**, `1,391` syn/neu, `91.2%` local LH); `LHCENT` (`42` neu, **95.2% I**, **`10,554` syn/neu**) | Fast hardwired/specialized **Tactical Hazard & Fortress Veto** branch running parallel to learned Mushroom Body memory and converging in `SMP/SLP/CRE`. |
| **Central Complex = Unified Midline Arena** | `CX` **84.8% Midline** (`CX_Tangential` **93.0% Mid, 91% I**, `CX_Ring` **97.6% Mid, 99.6% I**) | Both sides' strategic plans collide inside ONE shared midline recurrent workspace carved by inhibitory Tangential/Ring prophylaxis. |
| **PFL2 (Gain) vs PFL3 (Steering) Symmetry Split!** | **`PFL2`** (`12` neu, `2,380` syn/neu): **48.1% Ipsi / 49.8% Contra**; **`PFL3`** (`24` neu, `2,989` syn/neu): **0.0% Ipsi / 99.0% Contra** (`86.1%` → `LAL`) | **PFL2** computes bilateral **sharpness/urgency gain** (search depth extensions + adaptive tick count $T$); **PFL3** computes **100% contralateral opponent-modeling move steering** (bilinear goal alignment)! |
| **Top-Down Inhibitory Centrifugal Attention** | `524` neurons, **`5,251` syn/neu** (**71.8% I**), **41.7% Contra**, projecting **`ME 43.6% + LO 33.0%`** | Central Brain projects a heavy bilateral inhibitory mask straight back onto early 64-square board representations. |
| **Octopamine vs Serotonin Targets** | **Octopamine** (`216` neu) → `ME 32.3%, LO 8.9%, LAL 5.4%` (Eye & Steering arousal); **Serotonin** (`2,282` neu) → `SMP 7.0%, FB 5.2%` (Strategic Core persistence) | Two tiny global neuromodulators: **Octopamine** tunes tactical sharpness/clock gain on Stream A & Policy; **Serotonin** tunes long-horizon squeeze persistence & draw contempt in Stream B! |
