import pandas as pd
import numpy as np
import pyarrow.feather as feather
from pathlib import Path

ROOT = Path(r"C:\Users\vash\Desktop\stokcly\connectome_lab\data")
neurons = pd.read_csv(ROOT / "flywire_neurons.tsv", sep="\t", low_memory=False)
pre = feather.read_feather(ROOT / "neuropil_pre_783.feather")

# Columns are: pre_pt_root_id, neuropil, count
pre["neuropil"] = pre["neuropil"].astype(str)
pre["np_side"] = np.where(
    pre["neuropil"].str.endswith("_L"), "left",
    np.where(pre["neuropil"].str.endswith("_R"), "right", "center")
)
pre["base_np"] = pre["neuropil"].str.replace(r"_[LR]$", "", regex=True)

# Join with neuron metadata
m = pre.merge(
    neurons[["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "top_nt", "side"]],
    left_on="pre_pt_root_id", right_on="root_id", how="inner"
)

total_syn = m["count"].sum()
n_unique_neuropils = m["neuropil"].nunique()
print(f"==========================================================================")
print(f"  FLYWIRE v783 — 78-NEUROPIL WHOLE-BRAIN SYNAPTIC PROJECTION MATRIX")
print(f"==========================================================================")
print(f"Unique Proofread Neurons with Output Synapses : {m['root_id'].nunique():,}")
print(f"Distinct Neuropil Compartments                : {n_unique_neuropils}")
print(f"Total Mapped Pre-Synaptic Sites               : {total_syn:,.0f}")

# Per-neuron summary for accurate Syn/Neuron and Neuropil Fan-Out (100% vectorized!)
per_neu = m.groupby("root_id", as_index=False).agg(
    total_syn=("count", "sum"),
    super_class=("super_class", "first"),
    cell_class=("cell_class", "first"),
    cell_sub_class=("cell_sub_class", "first"),
    cell_type=("cell_type", "first"),
    top_nt=("top_nt", "first"),
    side=("side", "first"),
)

fan5 = m[m["count"] >= 5].groupby("root_id")["neuropil"].nunique().rename("fanout_5")
per_neu = per_neu.merge(fan5, on="root_id", how="left")
per_neu["fanout_5"] = per_neu["fanout_5"].fillna(0)

# Ipsilateral / Contralateral / Midline per row
m["flow_hemi"] = np.where(
    m["np_side"] == "center", "mid",
    np.where(
        (m["side"].isin(["left", "right"])) & (m["side"] == m["np_side"]), "ipsi",
        np.where((m["side"].isin(["left", "right"])) & (m["side"] != m["np_side"]), "contra", "other")
    )
)

def summarize_subset(name, mask_neu, mask_edge):
    sub_n = per_neu[mask_neu]
    sub_e = m[mask_edge]
    if len(sub_n) == 0:
        return
    syn = sub_e["count"].sum()
    spn = sub_n["total_syn"].mean()
    med = sub_n["total_syn"].median()
    fo = sub_n["fanout_5"].mean()
    hemi = sub_e.groupby("flow_hemi")["count"].sum()
    tot_h = hemi.get("ipsi", 0) + hemi.get("contra", 0) + hemi.get("mid", 0)
    ipsi = hemi.get("ipsi", 0) / tot_h * 100 if tot_h else 0.0
    contra = hemi.get("contra", 0) / tot_h * 100 if tot_h else 0.0
    mid = hemi.get("mid", 0) / tot_h * 100 if tot_h else 0.0
    print(f"{name:<22} {len(sub_n):>7,} {syn:>11,.0f} {syn/total_syn*100:>6.2f}% {spn:>8.1f} {med:>7.0f} {fo:>6.2f} {ipsi:>6.1f}% {contra:>7.1f}% {mid:>6.1f}%")

print("\n--- 1. TRUE SYNAPTIC COMPUTE BUDGET & CROSS-HEMISPHERE FLOW (BY SUPER_CLASS) ---")
print(f"{'super_class':<22} {'Neurons':>7} {'TotalSyn':>11} {'%Brain':>7} {'MeanSyn':>8} {'MedSyn':>7} {'FanOut':>6} {'Ipsi%':>7} {'Contra%':>8} {'Mid%':>7}")
for sc in per_neu.groupby("super_class")["total_syn"].sum().sort_values(ascending=False).index:
    summarize_subset(str(sc), per_neu["super_class"] == sc, m["super_class"] == sc)

print("\n--- 2. KEY BIOLOGICAL CIRCUITS — TRUE SYNAPTIC WEIGHT & CROSS-TALK ---")
print(f"{'Circuit':<22} {'Neurons':>7} {'TotalSyn':>11} {'%Brain':>7} {'MeanSyn':>8} {'MedSyn':>7} {'FanOut':>6} {'Ipsi%':>7} {'Contra%':>8} {'Mid%':>7}")
specs = [
    ("Kenyon_Cell (All)", lambda d: d["cell_class"] == "Kenyon_Cell"),
    ("  KC_gamma (Fast)", lambda d: d["cell_type"].fillna("").str.startswith("KCg")),
    ("  KC_ab (Frozen)", lambda d: d["cell_type"].fillna("").str.startswith("KCab")),
    ("  KC_apbp (Game)", lambda d: d["cell_type"].fillna("").str.startswith(("KCapbp", "KCa'b'"))),
    ("MBON (Readout)", lambda d: d["cell_class"] == "MBON"),
    ("DAN (All Dopamine)", lambda d: d["cell_class"] == "DAN"),
    ("  DAN_PAM (Reward)", lambda d: d["cell_type"].fillna("").str.startswith("PAM")),
    ("  DAN_PPL1 (Alarm)", lambda d: d["cell_type"].fillna("").str.startswith("PPL1")),
    ("APL (GABA Gate)", lambda d: d["cell_type"] == "APL"),
    ("DPM (Consolidator)", lambda d: d["cell_type"] == "DPM"),
    ("LHLN (LH Veto 98% I)", lambda d: d["cell_class"] == "LHLN"),
    ("LHCENT / LH Output", lambda d: d["cell_class"].fillna("").isin(["LHCENT", "LH"])),
    ("CX (All Central Cpx)", lambda d: d["cell_class"] == "CX"),
    ("  CX_Columnar (89% E)", lambda d: (d["cell_class"] == "CX") & (d["cell_sub_class"] == "columnar")),
    ("  CX_Tangential(91% I)", lambda d: (d["cell_class"] == "CX") & (d["cell_sub_class"] == "tangential")),
    ("  CX_Ring (99.6% I)", lambda d: (d["cell_class"] == "CX") & (d["cell_sub_class"] == "ring neuron")),
    ("  CX_PFL2 (Gain)", lambda d: d["cell_type"].fillna("").str.startswith("PFL2")),
    ("  CX_PFL3 (Steer)", lambda d: d["cell_type"].fillna("").str.startswith("PFL3")),
    ("T4/T5 (ON/OFF Rays)", lambda d: d["cell_type"].fillna("").str.startswith(("T4", "T5"))),
    ("Visual Centrifugal", lambda d: d["super_class"] == "visual_centrifugal"),
    ("Descending (Motor)", lambda d: d["super_class"] == "descending"),
    ("Octopamine (Arousal)", lambda d: d["top_nt"] == "octopamine"),
    ("Serotonin (Patience)", lambda d: d["top_nt"] == "serotonin"),
]
for name, fn in specs:
    summarize_subset(name, fn(per_neu), fn(m))

print("\n--- 3. EXACT OUTPUT PROJECTION HIGHWAYS (CIRCUIT -> TOP TARGET NEUROPILS) ---")
for name, fn in [
    ("visual_projection", lambda d: d["super_class"] == "visual_projection"),
    ("visual_centrifugal", lambda d: d["super_class"] == "visual_centrifugal"),
    ("Kenyon_Cell", lambda d: d["cell_class"] == "Kenyon_Cell"),
    ("MBON (Readout)", lambda d: d["cell_class"] == "MBON"),
    ("DAN_PAM (Reward)", lambda d: d["cell_type"].fillna("").str.startswith("PAM")),
    ("DAN_PPL1 (Alarm)", lambda d: d["cell_type"].fillna("").str.startswith("PPL1")),
    ("LHLN (Lateral Horn)", lambda d: d["cell_class"] == "LHLN"),
    ("CX_Columnar", lambda d: (d["cell_class"] == "CX") & (d["cell_sub_class"] == "columnar")),
    ("CX_Tangential", lambda d: (d["cell_class"] == "CX") & (d["cell_sub_class"] == "tangential")),
    ("CX_PFL2+PFL3", lambda d: d["cell_type"].fillna("").str.startswith(("PFL2", "PFL3"))),
    ("Descending (Motor)", lambda d: d["super_class"] == "descending"),
    ("Octopamine", lambda d: d["top_nt"] == "octopamine"),
    ("Serotonin", lambda d: d["top_nt"] == "serotonin"),
]:
    sub = m[fn(m)]
    tot = sub["count"].sum()
    top = sub.groupby("base_np")["count"].sum().sort_values(ascending=False).head(6)
    s = ", ".join(f"{r}:{v/tot*100:.1f}%" for r, v in top.items())
    print(f"  {name:<20} -> {s}")

print("\n--- 4. TOP 15 NEUROPILS BY TOTAL SYNAPTIC TRAFFIC ---")
top_np = m.groupby("base_np")["count"].sum().sort_values(ascending=False).head(15)
for r, v in top_np.items():
    print(f"  {r:<10}: {v:>12,.0f} ({v/total_syn*100:>5.2f}%)")
