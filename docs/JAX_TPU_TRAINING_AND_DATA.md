# SACCADION — منظومة البيانات الاستراتيجية والتدريب فائق السرعة على Colab Free TPU (`JAX + Flax NNX + Muon`)

> **المبدأ الأساسي:** التدريب التقليدي (`AdamW` + خسارة MSE على رقم تقييم وحيد من Stockfish) بطيء ويحبس سقف ذكاء المودل تحت Stockfish.
> **مهمة هذه الوثيقة:** تصميم منظومة بيانات استراتيجية تتجاوز سقف Stockfish، وبروتوكول تدريب عصبي مخصص بـ **Python + JAX + Flax NNX + `optax.contrib.muon`** يضاعف سرعة التعلّم بـ **5×–10×** على **Google Colab Free TPU (`v2-8` / `v5e-1`)** ويصدّر أوزانًا مكمّمة بدقة `INT8/INT16` مباشرة إلى محرك **Pure Rust (`.saccnet`)** دون خسارة نقطة Elo واحدة.

---

## 1. منظومة البيانات الاستراتيجية "ما بعد Stockfish" (الركائز الأربع)

لكي يبتكر **Saccadion** استراتيجيات أعمق وأعقد من Stockfish ويخنق الخصوم ببطء عبر 15–20 نقلة، لا نكتفي بتقطير أرقام Stockfish بل نبني **4 روافد بيانات متكاملة**:

### الركيزة 1 — تعدين النقاط العمياء (`Lc0 BT4/T80` ضد `Stockfish 18`)
- ندمج بيانات مفتوحة عالية الجودة من **مباريات تدريب Lc0** (تحمل سياسات MCTS عميقة وتوزيعات WDL غنية) + **قواعد تقييم Stockfish العميقة** + **مباريات النخبة (TCEC / CCC / Lichess Elite GM)**.
- **تضخيم الاختلاف الاستراتيجي (`Divergence Upsampling 3×`):** كل وضعية يختلف فيها تقييم Lc0 عن Stockfish بأكثر من `100 cp` (تضحيات جودة استراتيجية، اختناقات بيادق مغلقة، حصون، هجمات ألوان متعاكسة) تحصل على وزن أخذ عينات مضاعف `3×` وتُوسَم **بنتيجة المباراة الحقيقية طويلة المدى + التقييم العميق** لكسر النقاط العمياء في كليهما!

### الركيزة 2 — الاستخراج التلقائي لمسارات "الضغط الاستراتيجي" لـ 10–20 نقلة (بـ Pure Rust، تكلفة محرك = صفر!)
عندما يقرأ `crates/saccadion-data` أي مباراة كاملة (PGN أو `.binpack`)، لا ينظر للوضعية $s_k$ بمعزل عن مستقبلها؛ بل يستشرف **تلقائيًا** الوضعيات المستقبلية في نفس المباراة عند أنصاف النقلات $k+6$ و $k+12$ و $k+20$ ويحسب **5 أهداف ضغط استراتيجي كمّية دقيقة**:

| الهدف الاستراتيجي | الصيغة المحسوبة تلقائيًا في Rust | ماذا يعلّم دماغ Saccadion؟ |
| :--- | :--- | :--- |
| **1. انهيار حرية حركة الخصم (`Mobility Collapse`)** | $\Delta M = \frac{\text{SafeMob}_{\text{opp}}(k) - \text{SafeMob}_{\text{opp}}(k+12)}{\text{SafeMob}_{\text{opp}}(k) + 4}$ | يعلّم الـ Central Complex كيف **يخنق الخصم تدريجيًا** قبل ظهور أي مكسب مادي! |
| **2. تقلّص إنتروبيا النقلات (`Zugzwang / Forcing Squeeze`)** | انخفاض عدد النقلات الآمنة المتاحة للخصم دون خسارة فورية خلال 6–12 نصف نقلة | يعلّم صناعة **الضغط الإجباري و الـ Zugzwang** في الوسط والنهايات. |
| **3. تراكم ضغط حلقة الملك (`King-Zone Pressure Delta`)** | ارتفاع وزن القطع المهاجمة لمربعات حلقة ملك الخصم التسعة بين $k$ و $k+10$ | يعلّم بناء الهجمات الهادئة المتراكمة (`h4-h5`, `Rg3`, `Nf5`) قبل التضحية النهائية! |
| **4. كسب المساحة والمربعات القوية (`Space & Outpost Gain`)** | صافي السيطرة على نصف رقعة الخصم + البيادق السالكة + نقاط الارتكاز المحمية بعد 12 نصف نقلة | يعلّم **الضغط الموضعي الكلاسيكي العميق** الذي يميز كبار الأساتذة و Lc0. |
| **5. مقياس الوقاية (`Prophylaxis Score`)** | مقدار هبوط نشاط أفضل قطعة هجومية للخصم مباشرة بعد نقلة هادئة غير آكلة (`Kh1!`, `a4!`, `Bf1!`) | يعلّم **حس الوقاية الاستباقية** (تغذية مباشرة لخلايا `hDelta` ذات الإزاحة الطورية 180°!). |

### الركيزة 3 — حزمة تدريبات المفاهيم والحصون (`Concept Drills` — 10% من كل Epoch)
- **الحصون والتعادلات النظرية (`Fortress Slice`):** فيلة الألوان المتعاكسة، سلاسل البيادق المقفلة بالكامل، حصون الرخ ضد الوزير ← تدرب مباشرة بوابة **فيتو `Lateral Horn` (`veto_fortress`)**.
- **نقاط التحول الهادئة (`Quiet Strategic Inflections`):** وضعيات لا يتغير فيها الميزان المادي لـ 8 أنصاف نقلات متتالية، لكن احتمال الفوز يقفز بأكثر من `25%` ← تدرب **توجيه `PFL3`** و**الإبداع**.

### الركيزة 4 — التطور التنافسي (`Phase 5: Anti-Stockfish Co-Evolution & Self-Play RL`)
بعد تخطي مستوى GM/Super-GM من الركائز 1–3:
1. نُجري محليًا على CPU مباريات سريعة عبر `fastchess` (`Saccadion vs Saccadion` و `Saccadion vs Stockfish / Lc0`).
2. نلتقط تلقائيًا الوضعيات المفصلية (التي خسر فيها Saccadion أو تفوّق فيها بضغط مفاجئ) ونُعيد تقييمها بعمق عالٍ.
3. نُجري جولات ضبط دقيق سريعة (RL Fine-Tuning) على Colab TPU — مما يسمح للمودل **باكتشاف استراتيجيات مبتكرة لم يرها Stockfish ولا Lc0!**

---

## 2. صيغة ملفات التدريب الثنائية فائقة السرعة (`.sacc` Shards)

لكي لا يصبح القرص أو CPU عنق زجاجة أمام سرعة **TPU `bfloat16`** (التي تعالج > 100,000 وضعية/ثانية)، يحوّل `crates/saccadion-data` المباريات إلى شرائح ثنائية مضغوطة بترويسة موحّدة (**Magic: `SACC0001`**) وسجلات ثابتة الحجم (**128 بايت لكل وضعية محاذية للذاكرة**) مقسّمة إلى:
- **شرائح عشوائية موزونة بالدهشة (`random_shards/*.sacc`):** للمرحلتين 1 و 2 (`Batch = 4096–8192`).
- **شرائح نوافذ مسارات مترابطة (`traj_shards/*.sacc`):** سلاسل من **`K = 12` وضعية متتابعة من نفس المباراة** (`[B, 12]`) لتدريب **الدوبامين الحي وآثار الأهلية** في المرحلة 3 عبر `jax.lax.scan`!

### تخطيط السجل الثنائي (`SaccRecord` — 128 بايت بالضبط)
```
Offset   Size   Field                    Description
0..32    32 B   board_nibbles[32]        Packed 64 squares (4 bits/sq: empty + 12 piece types)
32..36    4 B   meta_flags               Side to move, castling, ep_file, halfmove, game_phase
36..48   12 B   wdl_target[3] (f32)      Soft teacher Win/Draw/Loss probabilities (sum = 1.0)
48..52    4 B   best_move_u16 + next_u16 Primary policy move (from<<6|to) + 2-ply VNC follow-up
52..68   16 B   top4_moves[4] + probs[4] Sparse Top-4 teacher policy moves (u16) + quantized probs (u16)
68..88   20 B   pressure_targets[5](f32) 10-20 ply: [mob_collapse, entropy, king_press, space, prophylaxis]
88..104  16 B   lh_targets[4] (f32)      [blunder_risk, fortress_prob, mate_danger, passer_urgency]
104..112  8 B   volatility + mlh (2xf32) PFL2 tactical surprise |V_deep - V_shallow| & moves-left
112..120  8 B   threat_mask_stm (u64)    Bitboard of squares attacked/pressured by STM (ThreatMap64)
120..128  8 B   threat_mask_opp (u64)    Bitboard of squares attacked/pressured by Opponent
```
- **المزايا:** قراءة فورية بـ `np.memmap` في Python/JAX بدون أي Parsing لـ FEN أثناء التدريب! وبسرعة > 1 GB/s من SSD / Colab RAM Disk!

---

## 3. هندسة التدريب في `JAX + Flax NNX` لـ Google Colab Free TPU

جميع ملفات التدريب تقع داخل `training/saccadion_tpu/` وتُطابق رياضيًا 100% محرك الاستدلال في `crates/saccadion-nn`:

```
training/saccadion_tpu/
├── config.py                # SaccadionConfig (Tiny / Nano / Pro) matching Rust byte-for-byte
├── layers.py                # FlyHash Claws, T4/T5 Shunting, APL 92%, LH Veto, Phasor CX, PFL2/3, STE-QAT
├── model.py                 # Dual-Stream SaccadionNet + Differentiable Eligibility Dopamine (lax.scan)
├── losses.py                # 8-Sensor Multi-Task Loss + JEPA VICReg + K=4 Orthogonality + B->A Distill
├── dataset.py               # Zero-copy .sacc binary loader & [B, K=12] trajectory batcher + prefetch
├── train.py                 # Muon+AdamW 3-Stage Neurodevelopmental Trainer + Colab Drive checkpointing
├── export_rust.py           # Quantized .saccnet binary exporter + Rust parity verification harness
└── Saccadion_Colab_TPU.ipynb# Ready-to-run Colab Free TPU notebook (v2-8 / v5e-1)
```

### التفاصيل الحسابية الحرجة لاستقرار `bfloat16` على TPU:
1. **استقرار التثبيط التقسيمي (`Shunting Inhibition`):** نستخدم `g_leak >= 0.5` (قابل للتعلّم ومقيد عبر `softplus + 0.5`) مع حساب القسمة بـ `float32` داخل نواة `bfloat16` لتفادي أي `NaN` أو انفجار تدرّج!
2. **قناع مخالب `FlyHash` الثابت/شبه الثابت:** يُولَّد بذرةً حتمية (`Deterministic PRNG Seed`) متطابقة تمامًا بين JAX و Rust، بحيث تمتلك كل خلية $\gamma$ و $\alpha'/\beta'$ بالضبط `8` مخالب متفرّقة منظمة!
3. **حساب المصفوفات في TPU:** كل الأبعاد من مضاعفات `64` و `128` (`64`, `128`, `256`, `512`, `768`, `1024`, `2048`) لتحقيق أقصى استفادة من مصفوفات **MXU 128×128** على TPU!

---

## 4. دالة الخسارة متعددة الحواس (8 أهداف متزامنة لاستخلاص أقصى ذكاء من كل وضعية)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{WDL}} + 0.60\,\mathcal{L}_{\text{PFL3-Policy}} + 0.40\,\mathcal{L}_{\text{Pressure}} + 0.30\,\mathcal{L}_{\text{Threat64}} + 0.20\,\mathcal{L}_{\text{PFL2-Vol}} + 0.15\,\mathcal{L}_{\text{LH-Veto}} + 0.25\,\mathcal{L}_{\text{JEPA}} + 0.10\,\mathcal{L}_{\text{Creativity}} + \lambda_{\text{apl}}\,\mathcal{L}_{\text{APL-Sparse}}$$

1. **$\mathcal{L}_{\text{WDL}}$ (تقييم كلا المسارين + التقطير الحي `B -> A`):**
   $$\mathcal{L}_{\text{WDL}} = \text{KL}(p_{\text{wdl}} \,\|\, \hat{p}_B) + \text{KL}(p_{\text{wdl}} \,\|\, \hat{p}_A) + 0.30\,\text{KL}\!\big(\operatorname{stop\_grad}(\hat{p}_B) \,\|\, \hat{p}_A\big)$$
   هذا التقطير المباشر يضغط حكمة المسار B التكراري داخل أوراق المسار A الخاطفة على CPU!
2. **$\mathcal{L}_{\text{PFL3-Policy}}$ (التقطير المتفرّق `Sparse Top-K` + تآزر نقلتين `VNC Synergy`):**
   خسارة Cross-Entropy على النقلات الأفضل (Top-4 Teacher Policy) لعدم هدر سعة الشبكة على الضوضاء في النقلات الميتة، بالإضافة إلى خسارة التنبؤ بالنقلة المتابعة (2-Ply Synergy).
3. **$\mathcal{L}_{\text{Pressure}}$ (توقّع الضغط الاستراتيجي لـ 10–20 نقلة):** خسارة Huber على المقادير الخمسة (انهيار الحركة، الإنتروبيا، ضغط الملك، المساحة، والوقاية).
4. **$\mathcal{L}_{\text{Threat64}}$ (بصر الفص البصري):** Binary Cross-Entropy على الـ 64 مربعًا (`threat_mask_stm` و `threat_mask_opp`) لتعليم الـ Optic Lobe الربط والهجمات خلال الساعة الأولى من التدريب!
5. **$\mathcal{L}_{\text{PFL2-Vol}}$ و $\mathcal{L}_{\text{LH-Veto}}$:** تدريب معامل الحدة (`PFL2`) وبوابات فيتو القرن الجانبي الأربعة (`Blunder`, `Fortress`, `Mate`, `Passer`).
6. **$\mathcal{L}_{\text{JEPA}}$ (المحاكاة الذهنية بدون شجرة + منع الانهيار `VICReg`):**
   يتنبأ `Central Complex` في الوضعية $s_k$ بالتمثيل الكامن المستقبلي $z_{k+2} = \operatorname{EMA\_Encoder}(s_{k+2})$ بعد نقلتين، مع حدّ تباين `Variance-Covariance (VICReg)` يمنع انهيار التمثيل إلى ثابت!
7. **$\mathcal{L}_{\text{Creativity}}$ (تعامد الخطط الأربعة $K=4$ وحداثة `MBON-α'3`):**
   $$\mathcal{L}_{\text{orth}} = \left\| \hat{G}\hat{G}^\top - I_4 \right\|_F^2$$
   يُجبر الخطط الأربعة في الـ Fan-Shaped Body على أن تكون **مختلفة استراتيجيًا بحق** لا نسخًا مكررة!
8. **$\mathcal{L}_{\text{APL-Sparse}}$ (تلدين تفرّق APL إلى 92%):**
   يدفع متوسط نسبة الخلايا غير الصفرية في خلايا Kenyon إلى الهدف المستهدف $\rho_{\text{target}} = 0.08$ (`8%` نشطة = `92%` أصفار دقيقة!).

---

## 5. المحسِّن `Muon + AdamW` وجدول التدريب التطوري (3 مراحل — أسرع 5×–10×)

### 5.1 المحسِّن الهجين (`optax.contrib.muon` + `AdamW`)
- **جميع المصفوفات الكثيفة ثنائية الأبعاد** (أشعة `T4/T5`، `Rich-Club`، `PFN/vDelta/hDelta/Delta7`، مخالب Calyx، ومخارج `PFL`): تُدرَّب بمحسِّن **Muon** (`5` تكرارات Newton-Schulz Orthogonalization في `bfloat16`) ← يمنع التحيز الطيفي (التركيز على المادة وإهمال البنية الموضعية) ويختصر خطوات التقارب بنحو **40–50%**!
- **جدول الإدخال المتفرّق + الانحيازات + LayerNorm + ثوابت APL و الدوبامين:** تُدرَّب بـ `AdamW` مع Warmup وجدول Cosine Decay.

### 5.2 المراحل التطورية الثلاث (يرقة ← شرنقة ← ذبابة بالغة)
1. **المرحلة 1 — اليرقة / تأسيس البصر السريع (`20%` من وقت TPU):**
   - $T = 2$ دورتان فقط في الـ Central Complex، Batch ضخم (`4096–8192`)، تركيز عالٍ على `Threat64 + WDL + Policy + LH-Veto`. سرعة مضاعفة 3× لبناء بصر شطرنجي حاد.
2. **المرحلة 2 — الشرنقة / التداول العميق والضغط الاستراتيجي (`50%` من وقت TPU):**
   - فتح الـ Central Complex إلى $T = 6$ دورات مع **إشراف عميق على كل دورة (Deep Supervision):**
     $$\mathcal{L}_{\text{CX}} = \sum_{t=1}^{6} \frac{t}{21}\, \mathcal{L}^{(t)}$$
     يضمن هذا أن كل دورة إضافية تحسّن الدقة فعليًا ويُعطي تدفق تدرجات مثاليًا، مع تفعيل `Pressure(10-20ply)` + `JEPA` + `K=4 Creativity`.
3. **المرحلة 3 — البلوغ / Meta-Plasticity الدوبامين + تفرّق APL 92% + تكميم `STE-QAT` (`30%` من وقت TPU):**
   - **Meta-Training للدوبامين وأثر الأهلية:** نُغذّي دفعات مسارات مترابطة `[B, K=12]` داخل `jax.lax.scan`؛ في كل خطوة $j \in \{0\dots 11\}$ يتنبأ المسار A باستخدام $\Delta W_\gamma^{(j)}$ الحالي، ثم يحسب خطأ المفاجأة $\delta_j$، ويُحدّث أثر الأهلية $e_\gamma^{(j)}$ و $\Delta W_\gamma^{(j+1)}$ **داخل الرسم التفاضلي لـ JAX**! هكذا تشتق TPU التدرج *عبر* قانون التكيّف نفسه ليتعلم الدماغ كيف يصحح نفسه في 1–3 خطوات بينما $W_{\text{slow}}$ جامد!
   - **تلدين تفرّق APL:** يُرفع الهدف تدريجيًا من `50%` ← `92%` أصفار دقيقة.
   - **التكميم الواعي أثناء التدريب (`STE-QAT`):** تفعيل محاكاة التكميم بـ Straight-Through Estimator (`INT16` لجدول المسار A و `INT8` للطبقات الكثيفة) في آخر 15% من الخطوات لتفادي أي تدهور عند التصدير لـ Rust!

---

## 6. استمرارية التدريب على Colab المجاني + صيغة التصدير `.saccnet`

1. **البقاء على Colab Free TPU بدون ضياع (جلسات 3–12 ساعة):**
   - حفظ تلقائي كل **15 دقيقة** (أو كل `N` خطوة) على **Google Drive / HuggingFace Hub** بصيغة Checkpoint خفيفة تشمل (الأوزان + حالة Muon/AdamW + رقم الخطوة + رقم الشريحة + بذرة الخلط) لاستكمال الجلسة التالية من نفس النقطة بضغطة زر واحدة.
2. **صيغة الملف الثنائي الموحّد `.saccnet` (`SACC_NET_V1`):**
   - ترويسة ثابتة بحجم 64 بايت (`Magic: "SACCNET1"`, `version`, `arch_tier`, كل الأبعاد الهندسية + بذرة `FlyHash` الحتمية + معاملات التكميم `scales` + تحقق `CRC32`).
   - متبوعة بكتل التنسورات المرتّبة (`Little-Endian`, محاذية على `64 bytes` لـ **AVX-512 / AVX2**!) بحيث يقرأها `crates/saccadion-nn` مباشرة بدون أي مكتبة خارجية أو تحويل!
   - **بوابة التحقق الصارمة:** `export_rust.py` يختبر التطابق الرقمي التام بين مخرجات **JAX** ومخرجات **Rust SIMD** على نفس مجموعة الوضعيات قبل اعتماد أي وزن!

---

## 7. مصفوفة التجارب العلمية (`Ablation Study` على `Nano` قبل `Pro`)

قبل صرف ساعات TPU طويلة على `Saccadion-1 Pro`، نُجري اختبارًا علميًا مضبوطًا على **`Saccadion-1 Nano`** بقياس Elo الحقيقي عبر `fastchess` (SPRT) على نفس عدد عقد/وقت الـ CPU لإثبات مكسب كل ابتكار:

| التجربة | الإضافة المختبَرة | معيار النجاح المستهدف |
| :--- | :--- | :--- |
| **Exp 0 (Baseline)** | Stream A قياسي (بدون APL 92%، بدون Stream B، بدون Dopamine) | خط أساس Elo و NPS |
| **Exp 1 (+ FlyHash & APL 92%)** | تفعيل مخالب FlyHash + تفرّق APL 92% بـ AVX2 NNZ | قفزة NPS `> 1.5×` مع الحفاظ على/رفع Elo |
| **Exp 2 (+ Eligibility Dopamine)** | تفعيل $\Delta W_\gamma + \Delta W_{\alpha'/\beta'}$ مع `PAM/12×PPL1` | `+25` إلى `+60 Elo` مقابل Baseline بدون أي تغيير في الأوزان الجامدة! |
| **Exp 3 (+ Lateral Horn Veto)** | تفعيل فيتو الحصون والأخطار الفطري | خفض بلندرات الأفق ورفع دقة الحصون بنسبة `> 40%` |
| **Exp 4 (+ Phasor Stream B & PFL2/3)** | تفعيل الدماغ العميق الانتقائي (`1–2%` عقد) + `StrategicTT` + `Efference Copy` | `+80` إلى `+150 Elo` في الوضعيات الاستراتيجية والمغلقة! |
| **Exp 5 (+ Strategic Pressure & K=4 Creativity)** | إضافة أهداف الضغط لـ 10–20 نقلة + JEPA + تعامد الخطط الأربعة | تفوّق نوعي واضح في ألعاب الخنق الطويل والتضحيات الموضعية ضد محركات Alpha-Beta التقليدية! |
| **Exp 6 (`Saccadion-2 Causal` + 50× FastCache + CPU Stockfish Arena)** | منهج المفاهيم السببية الـ 16 النخبوية (Lichess Puzzles `Rating >= 2000, Popularity >= 85`) + `Counterfactual Noduli` + `Dop1R1/Dop1R2` + `MBON->DAN SOC` + كاش ثنائي `FastCache` (~25 ثانية/إيبوك بدل 20 دقيقة!) + حلبة Stockfish بالخلفية على CPU | قفزة 50× في سرعة التدريب على Colab TPU + قياس **Real Elo حقيقي** ضد Stockfish مع حصاد أخطاء المحرك وتدريبه عليها آليًا! |

---

## 7. تسريع 50× على Colab TPU + حلبة Stockfish الحية على CPU بالخلفية

1. **حل عنق الزجاجة (`50× FastCache`):**
   - فك تشفير 1.2 مليون وضعية داخل حلقات Python GIL كان يستهلك `98%` من زمن الإيبوك (`1,226s` مقابل `23s` حساب TPU فعلي!).
   - الآن يقوم أمر Pure-Rust **`saccadion-data predecode`** (أو بديل Python متعدد العمليات `ProcessPoolExecutor` لتجاوز GIL) بفك التشفير **مرة واحدة فقط** إلى ملفات memmap مدمجة (`int16`/`float16`) على القرص المحلي السريع `/content/sacc_cache/` (~1.5 ثانية بـ Rust لمليون وضعية!)، لتنزل مدة كل إيبوك لاحق إلى **~25 ثانية فقط (~45,000 pos/s)**!
2. **حلبة Stockfish بالخلفية على أنوية CPU الخاملة (`cpu_arena_sparring.py`):**
   - أثناء تدريب TPU، تلعب أنوية CPU الخاملة مباريات حقيقية سريعة بمحرك **`saccadion` Pure-Rust UCI** (يعيد تحميل `saccadion_latest.saccnet` آليًا بعد كل إيبوك!) ضد سلّم متدرّج: `Baseline-900 -> Stockfish 1320 -> 1550 -> 1850 -> 2200 -> 2600 -> 3000+ Elo`.
   - تحسب **Real MLE Elo** مع فترة ثقة 95%، وتحصد تلقائيًا نقاط التحول والخسائر ضد Stockfish (مع ردود Stockfish القاتلة) في `sparring_hard.sacc` لإعادة حقنها في التدريب!
3. **لوحة القيادة الحية بسرعة 20 Hz (`live_ui.py`):**
   - واجهة HTML/SVG داكنة بدون وميض داخل Google Colab تتحدث حتى 20 مرة/ثانية (`DisplayHandle`) لعرض شريط التقدم، الـ Loss، دقة الشطرنج، F1 للمفاهيم السببية الـ 16، نسبة خمول خلايا كينيون `APL`، ونتائج حلبة Stockfish والـ Real Elo لحظة بلحظة!
