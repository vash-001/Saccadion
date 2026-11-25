use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Default)]
struct GroupStats {
    total: usize,
    left: usize,
    right: usize,
    center: usize,
    nt_counts: BTreeMap<String, usize>,
    cell_types: HashSet<String>,
    sub_classes: BTreeMap<String, usize>,
}

impl GroupStats {
    fn record(&mut self, side: &str, nt: &str, cell_type: &str, sub_class: &str) {
        self.total += 1;
        match side {
            "left" => self.left += 1,
            "right" => self.right += 1,
            "center" => self.center += 1,
            _ => {}
        }
        let nt_key = if nt.is_empty() { "unknown" } else { nt };
        *self.nt_counts.entry(nt_key.to_string()).or_insert(0) += 1;
        if !cell_type.is_empty() {
            self.cell_types.insert(cell_type.to_string());
        }
        if !sub_class.is_empty() {
            *self.sub_classes.entry(sub_class.to_string()).or_insert(0) += 1;
        }
    }

    fn nt_summary(&self) -> String {
        let mut pairs: Vec<(&String, &usize)> = self.nt_counts.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1));
        pairs
            .into_iter()
            .take(5)
            .map(|(k, v)| format!("{}:{:.1}%", k, (*v as f64 / self.total as f64) * 100.0))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn ei_ratio(&self) -> (f64, f64, f64) {
        // In Drosophila central brain:
        // ACh = Excitatory
        // GABA + Glutamate (via GluCl in central brain) = Inhibitory
        // DA + 5-HT + Oct = Neuromodulatory
        let ach = *self.nt_counts.get("acetylcholine").unwrap_or(&0) as f64;
        let gaba = *self.nt_counts.get("gaba").unwrap_or(&0) as f64;
        let glu = *self.nt_counts.get("glutamate").unwrap_or(&0) as f64;
        let da = *self.nt_counts.get("dopamine").unwrap_or(&0) as f64;
        let ser = *self.nt_counts.get("serotonin").unwrap_or(&0) as f64;
        let oct = *self.nt_counts.get("octopamine").unwrap_or(&0) as f64;
        let known = ach + gaba + glu + da + ser + oct;
        if known == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (
            (ach / known) * 100.0,
            ((gaba + glu) / known) * 100.0,
            ((da + ser + oct) / known) * 100.0,
        )
    }
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "connectome_lab/data/flywire_neurons.tsv".to_string());

    let file = File::open(Path::new(&path)).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {}", path, e);
        std::process::exit(1);
    });
    let mut reader = BufReader::new(file);

    let mut header_line = String::new();
    reader.read_line(&mut header_line).expect("Failed to read header");
    let headers: Vec<&str> = header_line.trim_end_matches(&['\r', '\n'][..]).split('\t').collect();
    let col = |name: &str| -> usize {
        headers
            .iter()
            .position(|&h| h == name)
            .unwrap_or_else(|| panic!("Missing column: {}", name))
    };

    let idx_flow = col("flow");
    let idx_super_class = col("super_class");
    let idx_cell_class = col("cell_class");
    let idx_cell_sub_class = col("cell_sub_class");
    let idx_cell_type = col("cell_type");
    let idx_top_nt = col("top_nt");
    let idx_top_nt_conf = col("top_nt_conf");
    let idx_side = col("side");

    let mut total_neurons: usize = 0;
    let mut global_nt: BTreeMap<String, usize> = BTreeMap::new();
    let mut global_sides: BTreeMap<String, usize> = BTreeMap::new();
    let mut global_types: HashSet<String> = HashSet::new();
    let mut by_flow: BTreeMap<String, GroupStats> = BTreeMap::new();
    let mut by_super_class: BTreeMap<String, GroupStats> = BTreeMap::new();
    let mut by_cell_class: BTreeMap<String, GroupStats> = BTreeMap::new();

    // Specialized subsystem collectors
    let mut kc_by_type: BTreeMap<String, usize> = BTreeMap::new();
    let mut mbon_stats = GroupStats::default();
    let mut mbon_types: BTreeMap<String, usize> = BTreeMap::new();
    let mut dan_stats = GroupStats::default();
    let mut dan_subtypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut apl_dpm: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut cx_stats = GroupStats::default();
    let mut cx_subclasses: BTreeMap<String, GroupStats> = BTreeMap::new();
    let mut t4_t5_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut lc_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut alpn_count: usize = 0;
    let mut nt_conf_sum = 0.0_f64;
    let mut nt_conf_n = 0_usize;

    // Cell-type population distribution (for biological channel-width vs spatial replication)
    let mut type_pop: HashMap<String, usize> = HashMap::new();

    // Deep Circuit Dissection (Lateral Horn, PFL Steering, Neuromodulators, Superior Protocerebrum)
    let mut lh_subclasses: BTreeMap<String, GroupStats> = BTreeMap::new();
    let mut pfl_counts: BTreeMap<String, GroupStats> = BTreeMap::new();
    let mut sp_counts: BTreeMap<String, GroupStats> = BTreeMap::new();
    let mut neuromod_by_sc: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

    let mut line = String::new();
    while reader.read_line(&mut line).expect("read error") > 0 {
        let fields: Vec<&str> = line.trim_end_matches(&['\r', '\n'][..]).split('\t').collect();
        if fields.len() <= idx_side {
            line.clear();
            continue;
        }
        total_neurons += 1;

        let flow = fields[idx_flow];
        let super_class = fields[idx_super_class];
        let cell_class = fields[idx_cell_class];
        let sub_class = fields[idx_cell_sub_class];
        let cell_type = fields[idx_cell_type];
        let nt = fields[idx_top_nt];
        let side = fields[idx_side];

        if let Ok(conf) = fields[idx_top_nt_conf].parse::<f64>() {
            nt_conf_sum += conf;
            nt_conf_n += 1;
        }

        let nt_key = if nt.is_empty() { "unknown" } else { nt };
        *global_nt.entry(nt_key.to_string()).or_insert(0) += 1;
        let side_key = if side.is_empty() { "unknown" } else { side };
        *global_sides.entry(side_key.to_string()).or_insert(0) += 1;

        if !cell_type.is_empty() {
            global_types.insert(cell_type.to_string());
            *type_pop.entry(cell_type.to_string()).or_insert(0) += 1;
        }

        by_flow
            .entry(if flow.is_empty() { "unknown" } else { flow }.to_string())
            .or_default()
            .record(side, nt, cell_type, sub_class);

        by_super_class
            .entry(if super_class.is_empty() { "unknown" } else { super_class }.to_string())
            .or_default()
            .record(side, nt, cell_type, sub_class);

        if !cell_class.is_empty() {
            by_cell_class
                .entry(cell_class.to_string())
                .or_default()
                .record(side, nt, cell_type, sub_class);
        }

        // Mushroom Body dissection
        if cell_class == "Kenyon_Cell" || cell_type.starts_with("KC") {
            let key = if !cell_type.is_empty() {
                cell_type.to_string()
            } else if !sub_class.is_empty() {
                sub_class.to_string()
            } else {
                "KC_unspecified".to_string()
            };
            *kc_by_type.entry(key).or_insert(0) += 1;
        }
        if cell_class == "MBON" || cell_type.starts_with("MBON") {
            mbon_stats.record(side, nt, cell_type, sub_class);
            *mbon_types.entry(cell_type.to_string()).or_insert(0) += 1;
        }
        if cell_class == "DAN" || cell_type.starts_with("PAM") || cell_type.starts_with("PPL1") || cell_type.starts_with("PPL2") {
            dan_stats.record(side, nt, cell_type, sub_class);
            let bucket = if cell_type.starts_with("PAM") {
                "PAM (Reward / Appetitive)"
            } else if cell_type.starts_with("PPL1") {
                "PPL1 (Punishment / Aversive)"
            } else if cell_type.starts_with("PPL2") {
                "PPL2 (Modulatory)"
            } else {
                "Other DAN"
            };
            *dan_subtypes.entry(bucket.to_string()).or_insert(0) += 1;
        }
        if cell_type == "APL" || cell_type == "DPM" || cell_class == "MBIN" {
            let e = apl_dpm.entry(format!("{} [{}]", cell_type, cell_class)).or_insert((0, nt.to_string()));
            e.0 += 1;
        }
        if cell_class == "ALPN" {
            alpn_count += 1;
        }

        // Central Complex dissection
        if cell_class == "CX" {
            cx_stats.record(side, nt, cell_type, sub_class);
            let sc = if sub_class.is_empty() { "unspecified" } else { sub_class };
            cx_subclasses
                .entry(sc.to_string())
                .or_default()
                .record(side, nt, cell_type, sub_class);
        }

        // Optic Lobe directional (T4/T5) & Lobula Columnar (LC) detectors
        if cell_type.starts_with("T4") || cell_type.starts_with("T5") {
            *t4_t5_counts.entry(cell_type.to_string()).or_insert(0) += 1;
        }
        if cell_type.starts_with("LC") && cell_type.len() <= 6 && cell_type.chars().nth(2).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            *lc_counts.entry(cell_type.to_string()).or_insert(0) += 1;
        }

        // Lateral Horn (Innate Veto / Reflex Circuit)
        if cell_class.starts_with("LH") || cell_type.starts_with("LH") {
            let bucket = if !cell_class.is_empty() { cell_class } else { "LH_other" };
            lh_subclasses.entry(bucket.to_string()).or_default().record(side, nt, cell_type, sub_class);
        }

        // PFL Goal-Vector Steering Neurons (PFL1/2/3, EPG, Delta7, hDelta, vDelta, PFN)
        for prefix in ["PFL1", "PFL2", "PFL3", "EPG", "Delta7", "hDelta", "vDelta", "PFN", "PEN", "FB"] {
            if cell_type.starts_with(prefix) {
                pfl_counts.entry(prefix.to_string()).or_default().record(side, nt, cell_type, sub_class);
                break;
            }
        }

        // Superior Protocerebrum Convergence Hubs (SMP / SIP / CRE / SLP / LAL)
        for prefix in ["SMP", "SIP", "CRE", "SLP", "LAL"] {
            if cell_type.starts_with(prefix) {
                sp_counts.entry(prefix.to_string()).or_default().record(side, nt, cell_type, sub_class);
                break;
            }
        }

        // Global Neuromodulators (Octopamine, Serotonin, Dopamine) across super_classes
        if matches!(nt, "octopamine" | "serotonin" | "dopamine") {
            *neuromod_by_sc
                .entry(nt.to_string())
                .or_default()
                .entry(super_class.to_string())
                .or_insert(0) += 1;
        }

        line.clear();
    }

    println!("==================================================================");
    println!("  SACCADION CONNECTOME LAB — FLYWIRE v783 WHOLE-BRAIN ANALYSIS");
    println!("==================================================================");
    println!("Total Proofread Neurons : {}", total_neurons);
    println!("Distinct Cell Types     : {}", global_types.len());
    println!(
        "Mean NT Prediction Conf : {:.3}",
        if nt_conf_n > 0 { nt_conf_sum / nt_conf_n as f64 } else { 0.0 }
    );
    println!("\n--- 1. BILATERAL SYMMETRY (HEMISPHERES) ---");
    for (side, count) in &global_sides {
        println!("  {:<12}: {:>7} ({:>5.2}%)", side, count, (*count as f64 / total_neurons as f64) * 100.0);
    }

    println!("\n--- 2. GLOBAL NEUROTRANSMITTER BUDGET ---");
    for (nt, count) in &global_nt {
        println!("  {:<15}: {:>7} ({:>5.2}%)", nt, count, (*count as f64 / total_neurons as f64) * 100.0);
    }

    println!("\n--- 3. INFORMATION FLOW (AFFERENT / INTRINSIC / EFFERENT) ---");
    for (flow, st) in &by_flow {
        let (e, i, m) = st.ei_ratio();
        println!(
            "  {:<12}: {:>7} ({:>5.2}%) | types={:>5} | E(ACh)={:>5.1}% I(GABA+Glu)={:>5.1}% Mod={:>4.1}% | L/R/C={}/{}/{}",
            flow,
            st.total,
            (st.total as f64 / total_neurons as f64) * 100.0,
            st.cell_types.len(),
            e, i, m,
            st.left, st.right, st.center
        );
    }

    println!("\n--- 4. BRAIN SUPER-CLASSES (MACRO ARCHITECTURE & E/I BALANCE) ---");
    let mut sc_vec: Vec<(&String, &GroupStats)> = by_super_class.iter().collect();
    sc_vec.sort_by(|a, b| b.1.total.cmp(&a.1.total));
    for (sc, st) in sc_vec {
        let (e, i, m) = st.ei_ratio();
        println!(
            "  {:<20}: {:>7} ({:>5.2}%) | types={:>5} | E={:>5.1}% I={:>5.1}% Mod={:>4.1}% | L/R={}/{} | {}",
            sc,
            st.total,
            (st.total as f64 / total_neurons as f64) * 100.0,
            st.cell_types.len(),
            e, i, m,
            st.left, st.right,
            st.nt_summary()
        );
    }

    println!("\n--- 5. TOP 20 CELL CLASSES ---");
    let mut cc_vec: Vec<(&String, &GroupStats)> = by_cell_class.iter().collect();
    cc_vec.sort_by(|a, b| b.1.total.cmp(&a.1.total));
    for (cc, st) in cc_vec.into_iter().take(20) {
        let (e, i, m) = st.ei_ratio();
        println!(
            "  {:<28}: {:>6} | types={:>4} | E={:>5.1}% I={:>5.1}% Mod={:>4.1}%",
            cc, st.total, st.cell_types.len(), e, i, m
        );
    }

    println!("\n--- 6. MUSHROOM BODY (CALYX) — DUAL-TIMESCALE MEMORY CIRCUIT ---");
    let kc_total: usize = kc_by_type.values().sum();
    let mut kc_gamma = 0usize;
    let mut kc_ab = 0usize;
    let mut kc_apbp = 0usize;
    let mut kc_other = 0usize;
    for (k, v) in &kc_by_type {
        if k.starts_with("KCg") {
            kc_gamma += v;
        } else if k.starts_with("KCab") {
            kc_ab += v;
        } else if k.starts_with("KCapbp") || k.starts_with("KCa'b'") {
            kc_apbp += v;
        } else {
            kc_other += v;
        }
    }
    println!("  ALPN Sensory Projection Inputs : {}", alpn_count);
    println!("  Total Kenyon Cells (KCs)       : {} (Expansion ratio vs ALPN = {:.1}x)", kc_total, kc_total as f64 / alpn_count.max(1) as f64);
    println!("    - Gamma lobe (Fast/Short)    : {} ({:.1}%)", kc_gamma, (kc_gamma as f64 / kc_total.max(1) as f64) * 100.0);
    println!("    - Alpha'/Beta' (Medium/Game) : {} ({:.1}%)", kc_apbp, (kc_apbp as f64 / kc_total.max(1) as f64) * 100.0);
    println!("    - Alpha/Beta (Long/Frozen)   : {} ({:.1}%)", kc_ab, (kc_ab as f64 / kc_total.max(1) as f64) * 100.0);
    if kc_other > 0 {
        println!("    - Other KC subtypes          : {}", kc_other);
    }
    for (k, v) in &kc_by_type {
        println!("      * {:<18}: {}", k, v);
    }
    let (me, mi, mm) = mbon_stats.ei_ratio();
    println!("  MBON Readout Neurons           : {} ({} types) | Compression KC->MBON = {:.1}:1 | E={:.1}% I={:.1}% Mod={:.1}% | {}",
        mbon_stats.total, mbon_types.len(), kc_total as f64 / mbon_stats.total.max(1) as f64, me, mi, mm, mbon_stats.nt_summary());
    println!("  DAN Dopamine Modulators        : {} ({} types) | {}", dan_stats.total, dan_stats.cell_types.len(), dan_stats.nt_summary());
    for (k, v) in &dan_subtypes {
        println!("      * {:<30}: {} ({:.1}%)", k, v, (*v as f64 / dan_stats.total.max(1) as f64) * 100.0);
    }
    println!("  APL / DPM / MBIN Gate Neurons  :");
    for (k, (cnt, nt)) in &apl_dpm {
        println!("      * {:<28}: {} (nt={})", k, cnt, nt);
    }

    println!("\n--- 7. CENTRAL COMPLEX (CX) — RECURRENT SPATIAL & ACTION ENGINE ---");
    let (ce, ci, cm) = cx_stats.ei_ratio();
    println!(
        "  Total CX Neurons: {} | distinct types={} | E={:.1}% I={:.1}% Mod={:.1}% | {}",
        cx_stats.total, cx_stats.cell_types.len(), ce, ci, cm, cx_stats.nt_summary()
    );
    let mut cx_vec: Vec<(&String, &GroupStats)> = cx_subclasses.iter().collect();
    cx_vec.sort_by(|a, b| b.1.total.cmp(&a.1.total));
    for (sc, st) in cx_vec {
        let (e, i, m) = st.ei_ratio();
        println!(
            "    - {:<24}: {:>5} | types={:>3} | E={:>5.1}% I={:>5.1}% Mod={:>4.1}% | {}",
            sc, st.total, st.cell_types.len(), e, i, m, st.nt_summary()
        );
    }

    println!("\n--- 8. OPTIC LOBE DIRECTIONAL (T4/T5) & LOBULA COLUMNAR (LC) DETECTORS ---");
    let t4t5_total: usize = t4_t5_counts.values().sum();
    println!("  T4/T5 Direction-Selective Neurons: {} across {} subtypes", t4t5_total, t4_t5_counts.len());
    for (k, v) in &t4_t5_counts {
        println!("    - {:<10}: {}", k, v);
    }
    let lc_total: usize = lc_counts.values().sum();
    println!("  Lobula Columnar (LC) Feature Detectors: {} neurons across {} distinct LC types", lc_total, lc_counts.len());

    println!("\n--- 9. CELL-TYPE POPULATION DISTRIBUTION (SPATIAL REPLICATION VS UNIQUE CHANNELS) ---");
    let mut pops: Vec<usize> = type_pop.values().copied().collect();
    pops.sort_unstable();
    let single_pair = pops.iter().filter(|&&c| c <= 2).count();
    let small_cluster = pops.iter().filter(|&&c| (3..=16).contains(&c)).count();
    let medium_array = pops.iter().filter(|&&c| (17..=128).contains(&c)).count();
    let large_columnar = pops.iter().filter(|&&c| c > 128).count();
    println!("  Types with 1-2 neurons   (Unique / Bilateral Singleton): {:>5} ({:.1}%)", single_pair, single_pair as f64 / pops.len().max(1) as f64 * 100.0);
    println!("  Types with 3-16 neurons  (Small Modular Clusters)      : {:>5} ({:.1}%)", small_cluster, small_cluster as f64 / pops.len().max(1) as f64 * 100.0);
    println!("  Types with 17-128 neurons(Medium Feature Arrays)       : {:>5} ({:.1}%)", medium_array, medium_array as f64 / pops.len().max(1) as f64 * 100.0);
    println!("  Types with >128 neurons  (Retinotopic Columnar Arrays) : {:>5} ({:.1}%)", large_columnar, large_columnar as f64 / pops.len().max(1) as f64 * 100.0);

    println!("\n--- 10. LATERAL HORN (INNATE SURVIVAL / VETO CIRCUIT) ---");
    for (k, st) in &lh_subclasses {
        let (e, i, m) = st.ei_ratio();
        println!("  {:<20}: {:>5} | types={:>3} | E={:>5.1}% I={:>5.1}% Mod={:>4.1}% | {}", k, st.total, st.cell_types.len(), e, i, m, st.nt_summary());
    }

    println!("\n--- 11. CENTRAL COMPLEX KEY MICRO-CIRCUITS (COMPASS, GOAL, PFL STEERING) ---");
    for (k, st) in &pfl_counts {
        let (e, i, m) = st.ei_ratio();
        println!("  {:<12}: {:>5} | types={:>3} | E={:>5.1}% I={:>5.1}% Mod={:>4.1}% | L/R={}/{} | {}", k, st.total, st.cell_types.len(), e, i, m, st.left, st.right, st.nt_summary());
    }

    println!("\n--- 12. CONVERGENCE & PRE-MOTOR HUBS (SMP / SIP / CRE / SLP / LAL) ---");
    for (k, st) in &sp_counts {
        let (e, i, m) = st.ei_ratio();
        println!("  {:<10}: {:>5} | types={:>3} | E={:>5.1}% I={:>5.1}% Mod={:>4.1}% | {}", k, st.total, st.cell_types.len(), e, i, m, st.nt_summary());
    }

    println!("\n--- 13. GLOBAL NEUROMODULATOR DISTRIBUTION (DOPAMINE / SEROTONIN / OCTOPAMINE) ---");
    for (nt, map) in &neuromod_by_sc {
        let mut items: Vec<(&String, &usize)> = map.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1));
        let summary = items.into_iter().take(5).map(|(k, v)| format!("{}:{}", k, v)).collect::<Vec<_>>().join(", ");
        println!("  {:<12}: {}", nt, summary);
    }
}
