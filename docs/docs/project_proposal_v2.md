
# RL-Guided Refinement of scRNA-seq Clustering Using GAG-Sulfation Domain Knowledge

## Abstract

Single-cell RNA sequencing (scRNA-seq) has revolutionized our ability to characterize cellular heterogeneity, yet identifying biologically meaningful cell subtypes remains challenging. Current clustering approaches rely on general-purpose unsupervised methods that may overlook functionally relevant subpopulations defined by specific molecular programs. Here, we propose a reinforcement learning (RL) framework that adaptively refines scRNA-seq clusters by integrating domain-specific biological knowledge—specifically, glycosaminoglycan (GAG) sulfation pathway expression signatures associated with perineuronal net (PNN) formation. We will develop the first Gymnasium-compatible RL environment for scRNA-seq cluster refinement, enabling sequential decision-making over clustering operations (split, merge, re-cluster) guided by rewards that balance standard clustering quality metrics with GAG-sulfation program coherence. Using human brain scRNA-seq data, we will test whether RL agents can discover PNN-associated interneuron subtypes that standard pipelines miss, and whether learned policies outperform hand-coded heuristics and traditional hyperparameter optimization. This work bridges machine learning and single-cell biology, providing a generalizable framework for incorporating biological priors into unsupervised analysis while addressing a critical gap: the lack of principled methods for biology-aware cluster refinement. Our open-source implementation will enable the community to apply modern RL techniques to single-cell analysis and adapt our framework to other domain-specific clustering challenges.

---

## 1. Scientific Rationale

### Research Question

**Can reinforcement learning discover better clustering refinement strategies than hand-coded heuristics by learning from experience across multiple episodes?**

Specifically: When guided by domain-specific reward signals (GAG-sulfation pathway coherence), can an RL agent identify biologically meaningful cell subtypes—such as PNN-associated interneuron populations—that traditional unsupervised clustering methods overlook?

### Central Hypothesis

**H1 (Superiority of RL)**: An RL agent trained to balance clustering quality and GAG-sulfation program coherence will achieve higher composite reward (quality + biological signal) than baseline methods including: (a) Leiden clustering with grid-searched resolution, (b) greedy split/merge heuristics, and (c) Bayesian hyperparameter optimization.

**H2 (Biological Discovery)**: RL-refined clusters will exhibit distinct GAG-sulfation expression profiles corresponding to known PNN+ vs. PNN- interneuron subtypes, validated through differential expression analysis and literature concordance with parvalbumin+ (PV+) basket cell biology.

**H3 (Generalization)**: Policies learned on one dataset will transfer to held-out datasets (different donors, brain regions), maintaining performance advantages over baselines after standard batch correction.

### Gap in Research

Current scRNA-seq analysis pipelines face three critical limitations:

1. **Lack of Biology-Aware Refinement**: Standard clustering (Leiden, Louvain) optimizes graph modularity or silhouette scores without incorporating domain knowledge. Biologists manually post-process clusters using marker genes, but this is subjective, labor-intensive, and not reproducible.

2. **No Principled Framework for Multi-Objective Optimization**: Biological clustering requires balancing competing objectives—internal validity (tight clusters), external validity (correspondence to known types), and biological coherence (expression of relevant gene programs). No existing method optimizes this composite objective systematically.

3. **Absence of RL Environments for Genomics**: While RL has been successfully applied to molecular design, protein folding, and drug discovery, **no framework exists for sequential clustering decisions in scRNA-seq data**. This prevents the application of modern RL techniques to a fundamental genomics task.

### Our Innovation

We address these gaps by:

- **Creating the first RL environment for scRNA-seq clustering** with biologically-motivated state/action/reward spaces
- **Integrating GAG-sulfation domain knowledge** as a soft constraint guiding cluster discovery toward functionally relevant subtypes
- **Framing cluster refinement as sequential decision-making**, enabling credit assignment across multi-step refinement trajectories (which greedy methods cannot achieve)
- **Providing open-source infrastructure** enabling the community to adapt our framework to other gene programs and cell types

This represents a **paradigm shift** from biology-agnostic unsupervised learning to **biology-guided reinforcement learning** for single-cell analysis.

---

## 2. Background and Significance

### Current State of scRNA-seq Clustering

Single-cell transcriptomics enables molecular characterization of cellular diversity at unprecedented resolution. The standard analysis pipeline involves:

1. **Quality control and normalization** (UMI counts → log-normalized expression)
2. **Dimensionality reduction** (PCA, scVI latent spaces)
3. **Graph construction** (k-NN graphs in latent space)
4. **Community detection** (Leiden/Louvain clustering)
5. **Manual annotation** (marker gene analysis, differential expression)

**Key tools**: Scanpy, Seurat, scVI-tools are widely adopted and work well for coarse cell type identification (e.g., separating neurons from glia).

### Limitations of Current Approaches

#### **Problem 1: Hyperparameter Sensitivity**

Clustering resolution is a critical hyperparameter that determines granularity:
- **Low resolution** → Biologically distinct subtypes merged
- **High resolution** → Artifactual over-splitting

Current practice:
- Try multiple resolutions, pick one that "looks right"
- No principled method for optimization
- Results vary across labs and analysts

#### **Problem 2: Biology-Agnostic Objectives**

Leiden/Louvain optimize graph modularity—a measure of community structure with **no biological meaning**. High modularity doesn't guarantee:
- Functional coherence (cells in a cluster share biological programs)
- Correspondence to known cell types
- Discovery of rare but important subtypes

#### **Problem 3: The "Resolution Paradox"**

Higher resolution improves separation of distinct types **but** also increases noise and artifacts. Standard methods cannot simultaneously optimize for:
- Global structure (major cell types separated)
- Local structure (biologically meaningful subtypes within types)

**Example**: PNN+ parvalbumin interneurons are functionally distinct from PNN- PV interneurons, but both express GAD1/GAD2 (pan-inhibitory markers) and PVALB. Standard clustering often lumps them together unless resolution is very high, which then over-splits other populations.

#### **Problem 4: Post-Hoc Manual Refinement**

Biologists address these issues through manual intervention:
- Inspect marker gene expression
- Split clusters that look heterogeneous
- Merge clusters that seem redundant
- Iterate until "it looks right"

This approach is:
- **Subjective**: Different analysts get different results
- **Non-reproducible**: Cannot be applied systematically to new datasets
- **Labor-intensive**: Requires domain expertise and trial-and-error
- **Not scalable**: As datasets grow (millions of cells), manual curation becomes infeasible

### Why GAG-Sulfation and PNNs?

We focus on **glycosaminoglycan (GAG) sulfation pathways** as our domain knowledge case study because:

#### **Biological Significance**

GAG sulfation enzymes (CHST*, HS*ST*, NDST*) modify extracellular matrix (ECM) components, particularly in the formation of **perineuronal nets (PNNs)**—specialized ECM structures that:
- Surround specific interneuron subtypes (primarily PV+ fast-spiking basket cells)
- Regulate synaptic plasticity and critical period closure
- Protect neurons from oxidative stress
- Are implicated in neurological disorders (schizophrenia, Alzheimer's, epilepsy)

#### **Clustering Challenge**

PNN+ vs. PNN- interneurons:
- Share core interneuron identity (GAD1/GAD2 expression)
- Often share subtype markers (PVALB, SST)
- Differ primarily in ECM/GAG-related gene expression
- Are frequently merged in standard analyses despite functional differences

#### **Clinical Relevance**

PNN dysregulation is implicated in:
- Neurodevelopmental disorders (autism spectrum disorders)
- Neurodegeneration (Alzheimer's disease—Aβ plaques disrupt PNNs)
- Psychiatric conditions (schizophrenia—reduced PNN density in cortex)
- Epilepsy (PNN degradation increases seizure susceptibility)

Better characterization of PNN-associated cell states could inform therapeutic strategies targeting ECM remodeling.

#### **Generalizable Framework**

While we use GAG-sulfation as proof-of-concept, our framework applies to any gene program:
- Metabolic pathways (glycolysis, OXPHOS) for identifying metabolic states
- Stress response (heat shock, UPR) for disease-associated subtypes
- Cell cycle genes for proliferative vs. quiescent cells
- Developmental programs (HOX genes, transcription factors)

### Current Research Landscape

**RL in biology** has focused on:
- Molecular design (generating molecules with desired properties)
- Protein structure prediction (AlphaFold uses RL-adjacent methods)
- Drug discovery (optimizing compound libraries)

**But not on data analysis tasks** like clustering, where human expertise is still dominant.

**Clustering innovations** have focused on:
- Better distance metrics (scVI, Harmony for batch correction)
- Hierarchical approaches (combining resolutions)
- Consensus clustering (aggregating multiple runs)

**But not on adaptive, biology-guided refinement** using sequential decision-making.

### Significance of This Work

This project is significant because it:

1. **Bridges AI and genomics**: Applies modern RL to a fundamental single-cell analysis challenge
2. **Addresses reproducibility**: Provides automated, deterministic alternative to manual cluster refinement
3. **Enables biology-guided discovery**: Shows how domain knowledge can guide unsupervised analysis
4. **Creates reusable infrastructure**: First RL environment for scRNA-seq, enabling future methods development
5. **Tackles clinically relevant biology**: Better characterization of PNN-associated states has therapeutic implications

---

## 3. Specific Aims

### **Aim 1: Develop a Gymnasium-Compatible RL Environment for scRNA-seq Cluster Refinement**

#### Why It Matters

No existing RL environment captures the scRNA-seq clustering task. Creating this infrastructure will:
- Enable application of modern RL algorithms to single-cell analysis
- Provide a standardized benchmark for comparing clustering methods
- Allow the community to build upon our framework

#### Methods

**Environment Design (ScClusterEnv)**:

- **State Space**: Fixed-size feature vector (~30-40 dimensions) encoding:
  - Global metrics: # clusters, mean cluster size, size entropy
  - Quality metrics: silhouette score, graph modularity, connectivity
  - Biology metrics: GAG pathway enrichment scores (per gene set), separation statistics
  - Progress: current step / max steps

- **Action Space**: Discrete(5):
  - `0`: Split worst cluster (lowest local silhouette)
  - `1`: Merge closest pair (by centroid distance)
  - `2`: Re-cluster all (resolution +0.1)
  - `3`: Re-cluster all (resolution -0.1)
  - `4`: Accept (terminate episode)

- **Reward Function**:
  ```
  R = α·Q_cluster + β·Q_GAG - δ·Penalty

  Q_cluster = 0.5·silhouette + 0.3·modularity + 0.2·balance
  Q_GAG = ANOVA_F(GAG_scores ~ cluster) + MI(cluster, GAG_profile)
  Penalty = degenerate_states (too many/few clusters, singletons)

  Weights: α=0.6, β=0.4, δ=1.0 (tunable)
  ```

- **Episode Structure**:
  - Reset: Start with simple Leiden clustering (resolution=0.5)
  - Max 15 steps per episode
  - Terminate on "Accept" action or max steps

**Technical Implementation**:
- Python + Gymnasium API
- Integration with Scanpy/AnnData ecosystem
- Caching of intermediate results (graph, embeddings) for efficiency
- Compatible with Stable-Baselines3 for RL training

#### Expected Outcomes

✅ Functional RL environment passing `check_env()` validation
✅ Documentation and API for community use
✅ Benchmark dataset and baseline performance metrics

#### Potential Pitfalls

⚠️ **Computational cost**: Reward computation (silhouette, enrichment) may be slow
- *Mitigation*: Cache embeddings, use approximate methods (subsampling for silhouette)

⚠️ **Reward hacking**: Agent may exploit loopholes in reward function
- *Mitigation*: Entropy regularization, held-out validation, manual inspection of solutions

⚠️ **State representation**: Fixed-size vector may lose information as # clusters varies
- *Mitigation*: Use summary statistics (mean, max, std) to aggregate per-cluster features

---

### **Aim 2: Train RL Agents to Discover GAG-Sulfation-Associated Interneuron Subtypes**

#### Why It Matters

Demonstrates that RL can:
- Learn effective refinement strategies from experience
- Integrate biological knowledge (GAG pathways) to guide discovery
- Identify functionally relevant subtypes missed by standard methods

#### Methods

**Training Setup**:

- **Algorithm**: Proximal Policy Optimization (PPO)
  - Chosen for sample efficiency and stability
  - Policy network: 2-layer MLP (256→128→actions)
  - Value network: Shared backbone + value head

- **Hyperparameters**:
  - Learning rate: 3e-4
  - Batch size: 64
  - n_steps: 2048 (per environment per update)
  - Entropy coefficient: 0.01 (exploration)
  - GAE lambda: 0.95 (advantage estimation)
  - 4 parallel environments (DummyVecEnv)

- **Training Duration**: 50,000 timesteps (~30-60 min on M5 Mac)

- **Hardware**: Apple M5 Mac with MPS acceleration

**Dataset**:
- **Primary**: Human prefrontal cortex scRNA-seq (Allen Brain Cell Types or Mathys et al. 2019)
- **Focus**: Inhibitory interneurons (GAD1+/GAD2+ cells), N~2,000-5,000 cells
- **Embedding**: Pre-trained scVI latent (30 dimensions) or published latents

**GAG Gene Sets** (curated from GO/Reactome/UniProt):
- CS biosynthesis (CSGALNACT1/2, CHSY1/3, CHPF/2)
- CS sulfation (CHST11-15)
- HS biosynthesis (EXT1/2, EXTL1-3)
- HS sulfation (NDST1/2, HS2ST1, HS6ST1-3, HS3ST1/2)
- Sulfate activation (PAPSS1/2, SLC35B2/3)
- PNN core (ACAN, BCAN, NCAN, VCAN, HAPLN1/2/4, TNC, TNR)

**Validation**:
- Differential expression: Marker genes for RL-discovered clusters
- Pathway enrichment: GO/Reactome terms (expect ECM, GAG metabolism, synaptic terms)
- Known markers: Concordance with PV+, SST+, VIP+ interneuron markers
- Literature: Compare to published PNN+ interneuron signatures

#### Expected Outcomes

✅ RL agent learns stable policy (reward increases over training)
✅ Discovers 3-5 interneuron subclusters with distinct GAG profiles
✅ At least one subcluster enriched for PNN core genes (ACAN, BCAN, etc.)
✅ DE analysis shows PNN+ cluster has known PV+ interneuron markers

#### Potential Pitfalls

⚠️ **Overfitting to training data**: Policy may not generalize
- *Mitigation*: Train/test split (hold out donors), validate on independent dataset

⚠️ **Local optima**: Agent may converge to suboptimal strategy
- *Mitigation*: Multiple training runs with different seeds, ensemble policies

⚠️ **GAG signal too weak**: Sulfation genes may not cleanly separate clusters
- *Mitigation*: Validate GAG separation in raw data first, tune β weight

---

### **Aim 3: Benchmark RL Against Alternative Clustering Refinement Methods**

#### Why It Matters

Must demonstrate that RL's computational cost is justified by superior performance. Critical for assessing whether RL should be adopted vs. simpler alternatives.

#### Methods

**Baseline Methods**:

1. **Leiden Grid Search**:
   - Test resolutions [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
   - Pick resolution maximizing composite reward
   - Represents "exhaustive hyperparameter tuning"

2. **Greedy Heuristic**:
   ```python
   # Iterative split-if-improves, merge-if-improves
   for i in range(10):
       - Split worst cluster if reward improves
       - Merge closest pair if reward improves
       - Stop if no improvement
   ```
   - One-step lookahead optimization
   - Represents "smart manual refinement"

3. **Bayesian Optimization**:
   - Optimize resolution + GAG weight using Gaussian Process
   - 20 trials with acquisition function
   - Represents "black-box hyperparameter optimization"

4. **Hierarchical on GAG Genes**:
   - Cluster using only GAG gene expression
   - Tests whether GAG signal alone is sufficient

**Comparison Metrics**:

| Metric | What It Measures | Goal |
|--------|------------------|------|
| Silhouette | Cluster compactness | Maximize |
| GAG Separation | ANOVA F-score for GAG enrichment | Maximize |
| Composite Reward | α·Quality + β·GAG | Maximize |
| # Clusters | Granularity | Reasonable (3-8 for interneurons) |
| Compute Time | Efficiency | Minimize |
| Marker Purity | Biological coherence | Maximize |

**Statistical Analysis**:
- Run RL 5 times (stochastic policy during inference)
- Report mean ± std for RL, single values for deterministic baselines
- Paired tests where appropriate (same data)

**Visualization**:
- Pareto frontier: Silhouette vs. GAG separation
- UMAP comparisons: Clusters colored by method
- GAG heatmaps: Genes × clusters for each method
- Sankey diagrams: How cells move between baseline and RL clusters

#### Expected Outcomes

**Best-case scenario**:
✅ RL achieves 15-20% higher composite reward than best baseline
✅ Discovers novel subcluster not found by any baseline
✅ Improved GAG separation without sacrificing silhouette score

**Moderate scenario**:
✅ RL matches best baseline on composite reward
✅ Achieves similar result with less hyperparameter tuning
✅ More robust across random initializations

**Acceptable scenario**:
✅ RL performs comparably to best baseline
✅ Provides automated framework reducing manual intervention
✅ Demonstrates feasibility of RL for genomics clustering

#### Potential Pitfalls

⚠️ **Simple baselines perform well**: Leiden grid search may be "good enough"
- *Mitigation*: This is a valid finding; emphasize framework's generalizability and automation

⚠️ **Unfair comparison**: Baselines have access to held-out test data when picking hyperparameters
- *Mitigation*: Use nested cross-validation for baselines

⚠️ **Compute time**: RL training takes much longer than baselines
- *Mitigation*: Amortize cost—once trained, policy applies quickly; consider transfer learning

---

## 4. Our Innovative Approach

### Novel Hypothesis

**Central Innovation**: Clustering refinement is fundamentally a **sequential decision-making problem** where the optimal action at each step depends on the current state AND the anticipated future states. This temporal credit assignment challenge is precisely what RL excels at, but hand-coded heuristics and grid search cannot capture.

**Conceptual Framework**:

Traditional clustering treats hyperparameter selection as a **one-shot optimization**:
```
Choose parameters → Cluster → Evaluate → Done
```

We reframe it as a **Markov Decision Process (MDP)**:
```
State → Action → New State → Reward → (repeat) → Terminal State
```

This enables:
- **Multi-step planning**: Early actions affect later possibilities
- **Credit assignment**: Learn which early decisions led to final success
- **Adaptive strategies**: Different data → different refinement trajectories

### Bridging Disciplines

This project uniquely integrates three fields:

#### **1. Reinforcement Learning (Computer Science)**

**Core concepts applied**:
- MDPs for sequential decision-making
- Policy gradient methods (PPO) for non-differentiable rewards
- Exploration-exploitation tradeoff via entropy regularization
- Transfer learning for cross-dataset generalization

**Novel contribution to RL**:
- First environment for genomics clustering
- Benchmark for biology-aware reward shaping
- Case study in discrete action spaces with expensive state transitions

#### **2. Single-Cell Genomics (Biology)**

**Core challenges addressed**:
- Cell type/state identification from scRNA-seq
- Balancing clustering granularity (resolution paradox)
- Incorporating biological knowledge into unsupervised analysis
- Reproducible, automated cluster annotation

**Novel contribution to genomics**:
- Principled framework for biology-guided clustering
- Method for systematic GAG-sulfation subtype discovery
- Reusable infrastructure for other gene programs

#### **3. Extracellular Matrix Biology (Neuroscience)**

**Core knowledge integrated**:
- GAG sulfation pathway enzymology
- PNN structure and function in cortical circuits
- PNN+ interneuron physiology and disease relevance
- Sulfation heterogeneity across interneuron subtypes

**Novel contribution to ECM biology**:
- Computational method for PNN-associated cell state identification
- Framework for linking sulfation genotypes to cellular phenotypes
- Potential for discovering novel PNN-related subtypes

### Why This Integration Is Powerful

**Traditional clustering** (Leiden/Louvain):
- ❌ Optimizes graph modularity (no biological meaning)
- ❌ Single-pass optimization (no iterative refinement)
- ❌ Ignores domain knowledge

**Manual refinement** (current practice):
- ⚠️ Incorporates domain knowledge (biologist intuition)
- ❌ Not systematic or reproducible
- ❌ Doesn't scale

**Our RL approach**:
- ✅ Optimizes biologically meaningful composite objective
- ✅ Systematic, multi-step refinement strategy
- ✅ Incorporates domain knowledge (GAG pathways in reward)
- ✅ Reproducible and automatable
- ✅ Learns from experience (credit assignment)

### Paradigm Shift

This work represents a shift from:

**Passive clustering** (accept algorithm output) → **Active refinement** (iteratively improve)

**Biology-agnostic** (modularity maximization) → **Biology-guided** (reward coherence of gene programs)

**Manual curation** (expert-driven) → **Learned policies** (data-driven automation)

### Broader Impact on Single-Cell Analysis

If successful, this framework enables:

1. **Other gene programs**: Replace GAG-sulfation with any pathway of interest
2. **Other cell types**: Apply to T cells (activation states), cancer cells (EMT programs), stem cells (differentiation trajectories)
3. **Multi-modal integration**: Extend state space to include ATAC-seq, protein, spatial data
4. **Temporal dynamics**: RNA velocity as additional reward signal
5. **Interactive analysis**: User-in-the-loop RL for semi-supervised refinement

This is not just a better clustering method—it's a **new paradigm** for incorporating biological knowledge into unsupervised single-cell analysis.

---

## 5. Experimental Design and Methods

### Overview

We will implement and evaluate our RL framework in three phases:

**Phase 1**: Environment development and validation (Aims 1)
**Phase 2**: RL training and biological validation (Aim 2)
**Phase 3**: Benchmark comparisons and analysis (Aim 3)

### Detailed Methodology

#### **5.1 Data Acquisition and Preprocessing**

**Primary Dataset**:
- **Source**: Allen Brain Cell Types database or Mathys et al. (2019) human brain aging study
- **Tissue**: Prefrontal cortex (dorsolateral or medial)
- **Technology**: 10x Chromium or similar droplet-based scRNA-seq
- **Expected size**: 50,000-100,000 cells total
- **Focus subset**: ~2,000-5,000 inhibitory interneurons

**Inclusion criteria**:
- Published dataset with cell type annotations
- UMI counts available (raw or processed)
- Multiple donors (for train/test split)
- Cortical region (where PNNs are well-characterized)

**Preprocessing pipeline** (standard Scanpy):
```python
1. Quality control:
   - Filter cells: 200 < n_genes < 5000, mito% < 20
   - Filter genes: expressed in ≥3 cells

2. Normalization:
   - Total count normalization (10,000 UMIs per cell)
   - Log1p transformation

3. Feature selection:
   - Highly variable genes (3,000 genes, Seurat v3 method)

4. Batch correction & embedding:
   - scVI (30 latent dimensions) OR
   - Pre-computed latents if available

5. Cell type filtering:
   - Extract GAD1+ and/or GAD2+ cells (inhibitory interneurons)
   - Verify with other markers (SLC32A1, DLX1/2)

6. Graph construction:
   - k-NN graph (k=15) on scVI latent
   - Connectivities computed via Scanpy
```

**Why this dataset**:
- **Brain tissue**: PNNs are well-characterized in cortex
- **Inhibitory interneurons**: Known PNN+ (PV+) vs. PNN- subtypes
- **Human data**: Clinically relevant, generalization target for methods
- **Public availability**: Reproducibility and community validation

**Alternative datasets** (if primary unavailable):
- Lake et al. (2018) - human brain snRNA-seq
- Hodge et al. (2019) - Allen Institute MTG atlas
- Synthetic data (Splatter) - for controlled validation

#### **5.2 GAG-Sulfation Gene Set Curation**

**Approach**: Compile from multiple authoritative sources

**Sources**:
1. **Gene Ontology** (GO:xxxx terms):
   - Glycosaminoglycan biosynthetic process (GO:0006024)
   - Chondroitin sulfate biosynthetic process (GO:0030206)
   - Heparan sulfate biosynthetic process (GO:0015012)
   - Keratan sulfate biosynthetic process (GO:0018146)

2. **Reactome pathways**:
   - HS-GAG biosynthesis (R-HSA-2024096)
   - CS-GAG biosynthesis (R-HSA-1793185)
   - Glycosaminoglycan metabolism (R-HSA-1630316)

3. **Manual curation** (from literature):
   - PNN core: ACAN, BCAN, NCAN, VCAN
   - Link proteins: HAPLN1, HAPLN2, HAPLN4
   - Tenascins: TNC, TNR
   - Sulfotransferases: CHST* family, HS*ST*, NDST*

**Gene set structure** (JSON format):
```json
{
  "CS_biosynthesis": {
    "genes": ["CSGALNACT1", "CSGALNACT2", "CHSY1", ...],
    "description": "Chondroitin sulfate backbone synthesis",
    "source": "GO:0030206"
  },
  "CS_sulfation": {
    "genes": ["CHST11", "CHST12", "CHST13", "CHST14", "CHST15"],
    "description": "Chondroitin 4-O and 6-O sulfation",
    "source": "Manual curation (UniProt)"
  },
  ...
}
```

**Validation**:
- Check expression in dataset (remove genes with <3 cells expressing)
- Literature review: Verify genes are known GAG/PNN components
- Positive controls: Known PNN markers should enrich in expected cell types

#### **5.3 RL Environment Implementation (ScClusterEnv)**

**State Representation** (~35 dimensions):

```python
State vector = [
    # Global (3)
    n_clusters / n_cells,  # Normalized cluster count
    mean_cluster_size / n_cells,
    cluster_size_entropy,  # H = -Σ p_i log(p_i)

    # Quality (3)
    silhouette_score,  # Mean silhouette on scVI latent [-1, 1]
    graph_modularity,  # [0, 1]
    cluster_balance,  # 1 - (std_size / mean_size)

    # GAG enrichment (7 x 4 = 28)
    # For each of 7 gene sets:
    mean_enrichment,  # Mean AUCell score across clusters
    max_enrichment,   # Max cluster enrichment
    F_statistic,      # ANOVA: enrichment ~ cluster
    mutual_info,      # MI(cluster, enrichment_score)

    # Progress (1)
    step / max_steps
]
```

**Action Space** (Discrete, 5 actions):

| Action | Description | Implementation |
|--------|-------------|----------------|
| 0 | Split worst cluster | Find cluster with lowest mean silhouette → run Leiden at higher resolution on subgraph |
| 1 | Merge closest pair | Compute cluster centroids → merge two with minimum Euclidean distance |
| 2 | Re-cluster (+0.1 res) | Run Leiden on full graph with `resolution += 0.1` |
| 3 | Re-cluster (-0.1 res) | Run Leiden on full graph with `resolution -= 0.1` |
| 4 | Accept | Terminate episode with current clustering |

**Reward Function**:

```python
def compute_reward(adata, gene_sets, alpha=0.6, beta=0.4, delta=1.0):
    # Clustering quality
    sil = silhouette_score(adata.obsm['X_scvi'], adata.obs['clusters'])
    mod = modularity(adata.uns['neighbors'], adata.obs['clusters'])
    bal = 1 - (cluster_size_std / cluster_size_mean)
    Q_cluster = 0.5*sil + 0.3*mod + 0.2*bal

    # GAG enrichment separation
    gag_scores = []
    for gene_set in gene_sets.values():
        enrichment = compute_aucell(adata, gene_set)  # Per-cell scores
        F_stat = f_oneway(*[enrichment[clusters==i] for i in unique_clusters])
        gag_scores.append(F_stat.statistic)
    Q_GAG = np.mean(gag_scores)

    # Penalties
    penalty = 0
    if n_clusters == 1 or n_clusters > 0.3*n_cells:
        penalty = 5
    penalty += n_singletons  # Clusters with <10 cells

    # Composite
    reward = alpha*Q_cluster + beta*Q_GAG - delta*penalty
    return reward
```

**Episode Structure**:
- **Reset**: Start with Leiden at resolution=0.5 on scVI latent
- **Max steps**: 15 actions per episode
- **Termination**: "Accept" action OR max steps reached
- **State preservation**: Cache neighbors graph, scVI latent (don't recompute)

**Engineering Optimizations**:
- Cache silhouette scores (only recompute for changed clusters)
- Vectorize AUCell computation
- Use sparse matrix operations for graph metrics
- Parallel environment support (DummyVecEnv)

#### **5.4 RL Training (PPO)**

**Algorithm**: Proximal Policy Optimization (Schulman et al., 2017)

**Rationale for PPO**:
- Sample-efficient (important since environment steps are expensive)
- Stable (clipped objective prevents destructive updates)
- Handles discrete actions well
- Industry standard with mature implementations

**Policy Network Architecture**:
```
Input: State vector (35 dims)
  ↓
Dense(256) + ReLU
  ↓
Dense(128) + ReLU
  ├→ Policy head: Dense(5) + Softmax → Action probabilities
  └→ Value head: Dense(1) → State value estimate
```

**Training Hyperparameters**:

```python
PPO(
    policy="MlpPolicy",
    env=vectorized_env,
    learning_rate=3e-4,
    n_steps=2048,           # Steps per environment before update
    batch_size=64,          # Minibatch size for optimization
    n_epochs=10,            # Optimization epochs per update
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE advantage estimation
    clip_range=0.2,         # PPO clipping parameter
    clip_range_vf=None,     # No value function clipping
    ent_coef=0.01,          # Entropy regularization (exploration)
    vf_coef=0.5,            # Value function loss coefficient
    max_grad_norm=0.5,      # Gradient clipping
    device="mps",           # Apple Silicon acceleration
    seed=42,                # Reproducibility
    verbose=1
)
```

**Training Setup**:
- **Parallel environments**: 4 (via DummyVecEnv)
- **Total timesteps**: 50,000 (≈12,500 parameter updates)
- **Expected duration**: 30-60 minutes on M5 Mac
- **Checkpointing**: Save every 10,000 steps + best model (by mean reward)
- **Logging**: TensorBoard (reward, episode length, entropy, loss curves)

**Hardware**:
- Apple M5 MacBook Pro with MPS (Metal Performance Shaders)
- 16GB RAM minimum
- Fallback to CPU if MPS unavailable (acceptable, ~2x slower)

**Hyperparameter Tuning Strategy**:
- Start with default PPO hyperparameters
- If training unstable: reduce learning_rate to 1e-4
- If exploration insufficient: increase ent_coef to 0.05
- If reward hacking observed: increase penalty weight δ
- Monitor: reward curves, action distribution, episode lengths

#### **5.5 Baseline Methods Implementation**

**Baseline 1: Leiden Grid Search**
```python
resolutions = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
results = []
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added='clusters')
    reward = compute_reward(adata, gene_sets)
    results.append({'resolution': res, 'reward': reward, ...})
best = max(results, key=lambda x: x['reward'])
```

**Baseline 2: Greedy Heuristic**
```python
def greedy_refine(adata, gene_sets, max_iter=10):
    for i in range(max_iter):
        current_reward = compute_reward(adata, gene_sets)

        # Try split
        worst_cluster = find_worst_cluster(adata)
        adata_split = split_cluster(adata.copy(), worst_cluster)
        split_reward = compute_reward(adata_split, gene_sets)

        # Try merge
        c1, c2 = find_closest_clusters(adata)
        adata_merge = merge_clusters(adata.copy(), c1, c2)
        merge_reward = compute_reward(adata_merge, gene_sets)

        # Keep best
        if split_reward > max(current_reward, merge_reward):
            adata = adata_split
        elif merge_reward > current_reward:
            adata = adata_merge
        else:
            break  # No improvement, stop
    return adata
```

**Baseline 3: Bayesian Optimization**
```python
from skopt import gp_minimize
from skopt.space import Real

def objective(params):
    resolution, gag_weight = params
    sc.tl.leiden(adata, resolution=resolution)
    reward = compute_reward(adata, gene_sets, beta=gag_weight)
    return -reward  # Minimize negative reward

result = gp_minimize(
    objective,
    dimensions=[Real(0.2, 2.5, name='resolution'),
                Real(0.0, 1.0, name='gag_weight')],
    n_calls=20,
    random_state=42
)
```

**Baseline 4: Hierarchical on GAG Genes**
```python
# Extract mean GAG enrichment per cell
gag_matrix = compute_gag_enrichment_matrix(adata, gene_sets)  # cells × gene_sets

# Hierarchical clustering
from scipy.cluster.hierarchy import linkage, fcluster
Z = linkage(gag_matrix, method='ward')
clusters = fcluster(Z, t=5, criterion='maxclust')
adata.obs['gag_clusters'] = clusters
```

#### **5.6 Evaluation and Validation**

**Quantitative Metrics** (computed for all methods):

| Category | Metric | Formula/Method |
|----------|--------|----------------|
| **Clustering Quality** | Silhouette score | `silhouette_score(X_scvi, labels)` |
| | Graph modularity | Newman-Girvan modularity on k-NN graph |
| | Cluster balance | `1 - std(sizes) / mean(sizes)` |
| **GAG Signal** | ANOVA F-statistic | Mean across gene sets: `f_oneway(enrichment ~ cluster)` |
| | Mutual information | `mutual_info_score(clusters, gag_enrichment_bins)` |
| **Composite** | Weighted reward | `0.6*Q_cluster + 0.4*Q_GAG - penalties` |
| **Biological** | Marker purity | Fraction of cells with expected markers per cluster |
| | DE gene count | # significant genes (FDR < 0.05) per cluster |
| **Efficiency** | Compute time | Wall-clock time (seconds) |
| | # Iterations | Actions taken before convergence |

**Statistical Testing**:
- RL: 5 independent runs → report mean ± std
- Baselines: Deterministic → single values
- Pairwise comparisons: Wilcoxon signed-rank test (if paired) or Mann-Whitney U
- Multiple testing correction: Bonferroni or FDR

**Biological Validation**:

1. **Differential Expression Analysis**:
   ```python
   # For each method's clusters
   sc.tl.rank_genes_groups(adata, groupby='clusters', method='wilcoxon')

   # Extract top 50 markers per cluster
   markers = sc.get.rank_genes_groups_df(adata, group=None)
   markers = markers[markers['pvals_adj'] < 0.05]
   ```

2. **Pathway Enrichment** (using Enrichr or GSEA):
   - Input: Top 100 DE genes per cluster
   - Databases: GO Biological Process, Reactome, KEGG
   - Expected enrichment: ECM organization, GAG metabolism, synaptic terms

3. **Known Marker Concordance**:
   ```python
   # Check enrichment of known markers
   markers_dict = {
       'PV_interneurons': ['PVALB', 'GAD1', 'GAD2'],
       'SST_interneurons': ['SST', 'GAD1', 'GAD2'],
       'VIP_interneurons': ['VIP', 'GAD1', 'GAD2'],
       'PNN_associated': ['ACAN', 'BCAN', 'NCAN', 'HAPLN1']
   }

   # For each cluster, compute mean expression of marker sets
   # Expect: some clusters high PV+PNN, others high PV+low-PNN
   ```

4. **Literature Comparison**:
   - Compare to published PNN+ interneuron signatures (Fawcett et al., Carulli et al.)
   - Check for concordance with spatial transcriptomics (if available)
   - Validate with ISH data from Allen Brain Atlas

**Visualization Suite**:

1. **UMAP Comparisons**:
   - Side-by-side UMAPs colored by: Leiden baseline, RL clusters
   - Overlay: GAG gene expression, known markers (PVALB, ACAN)

2. **GAG Heatmaps**:
   - Rows: GAG genes (grouped by pathway)
   - Columns: Clusters (from each method)
   - Values: Mean scaled expression
   - Dendrogram: Hierarchical clustering of clusters by GAG profile

3. **Metric Comparisons**:
   - Bar plots: Each method's performance on each metric
   - Pareto frontier: Silhouette vs. GAG separation scatter
   - Radar plots: Multi-dimensional comparison across metrics

4. **Action Analysis** (RL-specific):
   - Action frequency: Which actions does agent prefer?
   - Episode trajectories: Reward over steps for sample episodes
   - State-action heatmaps: Which states trigger which actions?

5. **Cluster Transitions**:
   - Sankey diagram: How cells move from baseline → RL clusters
   - Confusion matrix: Overlap between method's cluster assignments

#### **5.7 Transfer Learning Evaluation** (Optional, time permitting)

**Goal**: Test if learned policy generalizes to new datasets

**Approach**:
1. Train on Dataset A (e.g., prefrontal cortex, donor 1-3)
2. Evaluate on Dataset B (e.g., prefrontal cortex, donor 4-5)
3. Evaluate on Dataset C (e.g., visual cortex or hippocampus)

**Expected outcome**:
- Perfect transfer: Unlikely (different cell type compositions)
- Partial transfer: Policy still beats baselines, but lower margin
- Fine-tuning: Brief re-training on new dataset improves performance

**Metrics**:
- Zero-shot performance: Apply trained policy directly
- Few-shot adaptation: Re-train for 5,000 steps on new data
- Full re-training: Train from scratch on new data

This tests whether RL learns generalizable refinement strategies vs. dataset-specific hacks.

---

## 6. Is This Project Feasible?

### Technical Feasibility: **HIGH** ✅

**Infrastructure exists**:
- ✅ Gymnasium API: Standard, well-documented
- ✅ Stable-Baselines3: Mature PPO implementation
- ✅ Scanpy/scVI: Industry-standard single-cell tools
- ✅ Public datasets: Allen Brain, Mathys et al. freely available
- ✅ Hardware: M5 Mac sufficient (MPS acceleration)

**Scope is manageable**:
- ✅ Discrete action space (5 actions) → easier than continuous control
- ✅ Moderate state space (~35 dims) → standard MLP handles easily
- ✅ Small dataset (2k-5k cells) → fast environment steps
- ✅ Short episodes (15 steps) → quick training
- ✅ 50k timesteps → reasonable training time (30-60 min)

**Fallback options**:
- If RL training unstable → Use A2C (simpler algorithm)
- If GAG signal weak → Use other pathways (stress response, metabolism)
- If dataset unavailable → Use synthetic data (Splatter) for validation
- If MPS issues → CPU training still viable (2x slower, acceptable)

### Biological Feasibility: **MODERATE-HIGH** ✅

**Known biology**:
- ✅ PNN+ interneurons are well-characterized
- ✅ GAG-sulfation pathways are well-annotated
- ✅ scRNA-seq can detect sulfation enzyme expression
- ⚠️ Low abundance transcripts may be noisy (mitigation: use gene sets, not individual genes)

**Validation path**:
- ✅ Known markers (PVALB, ACAN) provide positive controls
- ✅ Published signatures available for comparison
- ✅ Allen Brain ISH atlas for independent validation
- ⚠️ May not discover completely novel biology (but methods novelty is sufficient)

**Risk**: GAG genes may not cleanly separate clusters
- **Likelihood**: LOW—literature shows PNN+/PNN- interneurons have distinct sulfation profiles
- **Mitigation**: Validate GAG separation in raw data before RL training
- **Fallback**: Adjust β weight, try different gene sets, or change biological focus

### Computational Feasibility: **HIGH** ✅

**Resource requirements (estimated)**:

| Task | Time | Hardware |
|------|------|----------|
| Data download/preprocessing | 30 min | Mac M5, 8GB RAM |
| Gene set curation | 2 hours | Manual |
| Environment implementation | 8-12 hours | Coding |
| RL training (single run) | 45 min | Mac M5 with MPS |
| Baseline comparisons | 1 hour | Mac M5 |
| Evaluation & visualization | 4-6 hours | Coding + analysis |
| Documentation | 4-6 hours | Writing |

**Total**: ~25-35 hours over 1-2 weeks

**With coding agent (Claude Code)**:
- Environment implementation: **50% faster** (guided, debugged code)
- Baseline methods: **60% faster** (template-based)
- Visualization: **40% faster** (plotting boilerplate automated)

**Realistic timeline with Claude Code**: 15-25 hours over 1 week

### Reproducibility: **HIGH** ✅

**Deterministic components**:
- ✅ Random seeds set throughout (numpy, torch, scanpy)
- ✅ Frozen requirements (exact package versions)
- ✅ Public datasets with accession numbers
- ✅ Open-source code (GitHub)

**Stochastic components**:
- ⚠️ RL training (different runs → slightly different policies)
- **Mitigation**: Report mean ± std over 5 runs, provide best checkpoint

**Portability**:
- ✅ Works on Mac (MPS), Linux (CUDA), or CPU
- ✅ Conda environment ensures dependency consistency
- ✅ Docker option for ultimate reproducibility

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| RL doesn't converge | LOW | HIGH | Use proven PPO hyperparameters, monitor training |
| RL doesn't beat baselines | MODERATE | MODERATE | Valid finding; emphasize framework contribution |
| GAG signal insufficient | LOW | MODERATE | Validate upfront; try alternative pathways |
| Computational bottleneck | LOW | LOW | Optimize reward computation; use caching |
| Dataset unavailable | LOW | MODERATE | Multiple backup datasets identified |
| Code bugs | MODERATE | LOW | Unit tests; phased implementation |

**Overall feasibility**: **85-90%** (Very likely to succeed)

---

## 7. Expected Results and Impact

### Primary Expected Outcomes

#### **Outcome 1: Functional RL Environment**

**Deliverable**: Open-source `scrna-cluster-gym` package
- Gymnasium-compatible environment for scRNA-seq clustering
- Extensible to other gene programs and cell types
- Documentation, tutorials, example notebooks
- First RL benchmark for single-cell analysis

**Impact**:
- Enables community to test new RL algorithms on genomics tasks
- Provides standardized evaluation framework
- Catalyzes ML-genomics collaborations

#### **Outcome 2: Performance Demonstration**

**Best-case** (60% probability):
- RL achieves **15-25% higher composite reward** than best baseline
- Discovers **1-2 novel subclusters** with distinct GAG profiles
- One subcluster strongly enriched for PNN markers (ACAN, BCAN, HAPLN1)
- DE analysis confirms biological coherence (ECM/synaptic genes)

**Expected case** (90% probability):
- RL achieves **5-15% higher composite reward** OR matches best baseline
- Identifies known PNN+/PNN- interneuron split
- Demonstrates automated refinement (no manual hyperparameter tuning)
- Provides interpretable action sequences

**Acceptable case** (99% probability):
- RL performs comparably to sophisticated baselines
- Framework successfully integrates biological knowledge
- Demonstrates feasibility of RL for clustering
- Identifies limitations and future directions

#### **Outcome 3: Biological Insights**

**Interneuron subtype characterization**:
- Refined clusters correspond to known subtypes (PV+, SST+, VIP+)
- GAG-high cluster enriched for:
  - Core PNN components (ACAN, BCAN, NCAN, VCAN)
  - Sulfotransferases (CHST family, HS*ST*)
  - Synaptic genes (fast-spiking basket cell markers)
- GAG-low cluster shows alternative ECM programs

**Novel biology** (possible):
- Identification of intermediate PNN states (partial sulfation)
- Discovery of non-canonical PNN-associated genes
- Correlation with disease-associated gene signatures

**Validation**:
- Concordance with published PNN+ interneuron signatures (>70% overlap)
- Spatial validation using Allen Brain ISH atlas
- Literature support for identified marker combinations

### Impact on the Field

#### **Computational Biology & Genomics**

**Immediate impact**:
- **New methodology**: First demonstration of RL for scRNA-seq clustering
- **Open-source tools**: Reusable environment and training pipeline
- **Benchmark dataset**: Standard testbed for comparing clustering methods

**Long-term impact**:
- Shift toward **biology-guided machine learning** in genomics
- Template for integrating domain knowledge into unsupervised analysis
- Inspiration for RL applications to other genomics problems:
  - Trajectory inference (differentiation paths)
  - Spatial transcriptomics (region segmentation)
  - Multi-modal integration (ATAC+RNA)

#### **Neuroscience & ECM Biology**

**Immediate impact**:
- Better characterization of PNN-associated cell states
- Computational tool for analyzing GAG-sulfation heterogeneity
- Hypothesis generation: novel PNN subtypes for experimental follow-up

**Long-term impact**:
- Improved understanding of PNN diversity in circuits
- Links between sulfation genotype and neuronal phenotype
- Potential therapeutic targets (ECM remodeling in disease)

#### **Machine Learning**

**Immediate impact**:
- Novel RL environment for biology (expands RL application domains)
- Case study in reward shaping for scientific discovery
- Demonstration of discrete action spaces with expensive transitions

**Long-term impact**:
- Blueprint for **scientist-in-the-loop RL** (human expertise encoded in rewards)
- Advances in **multi-objective RL** (balancing competing goals)
- Inspiration for RL in other scientific domains (materials, chemistry, physics)

### Knowledge Gained

**Methodological insights**:
- When does RL outperform simpler alternatives for clustering?
- How sensitive is performance to reward function design?
- What is the optimal balance between exploration and biological coherence?
- Can policies transfer across datasets/tissues?

**Biological insights**:
- How heterogeneous is GAG-sulfation expression in cortical interneurons?
- Do sulfation profiles correlate with other functional markers?
- Are there discrete GAG-sulfation states or a continuum?
- Which sulfation enzymes are most informative for subtype identification?

**Computational insights**:
- What state representations best capture clustering quality?
- Which actions are most effective for refinement?
- How many steps are needed to converge to good solutions?
- What drives agent action selection (interpretability)?

### Publications and Dissemination

**Target journals**:
- **Methods**: *Nature Methods*, *Genome Biology*, *Bioinformatics*
- **Application**: *Nature Neuroscience*, *Neuron*, *Journal of Neuroscience*
- **ML**: *NeurIPS* (workshop), *ICML* (workshop on ML for biology)

**Conference presentations**:
- Single Cell Genomics (Wellcome Genome Campus)
- RECOMB (computational biology)
- Society for Neuroscience (if biological findings strong)

**Open-source release**:
- GitHub: Full code, documentation, tutorials
- PyPI: `pip install scrna-cluster-gym`
- Documentation site: API reference, examples, FAQs
- Preprint: bioRxiv upon completion

---

## 8. Translational Relevance

### Clinical Applications

#### **Neurological Disease Biomarkers**

**PNN dysregulation is implicated in**:
- Alzheimer's disease: Aβ plaques disrupt PNNs → circuit dysfunction
- Schizophrenia: Reduced PNN density in cortex → GABAergic deficits
- Epilepsy: PNN degradation → increased excitability
- Autism spectrum disorders: Altered PNN timing → critical period abnormalities

**Our framework enables**:
- Identification of disease-associated GAG-sulfation states in patient samples
- Comparison of PNN+ interneuron signatures across disease vs. control
- Discovery of biomarkers for patient stratification
- Targets for therapeutic intervention (ECM remodeling enzymes)

**Example translational path**:
```
1. Apply RL clustering to disease scRNA-seq → identify altered PNN states
2. Validate with spatial transcriptomics → localize to circuit regions
3. Test in mouse models → causal role of sulfation changes
4. Drug screening → compounds restoring normal PNN profiles
```

#### **Therapeutic Target Identification**

**Druggable targets from GAG pathways**:
- **Sulfotransferases** (CHST family): Small molecule inhibitors exist
- **Heparanase**: Enzyme degrading HSPGs, already targeted in cancer
- **Chondroitinase ABC**: Experimental PNN-digesting enzyme (stroke, spinal cord injury trials)

**Our method helps**:
- Prioritize which enzymes to target (most dysregulated in disease)
- Predict cell type-specific effects (avoid off-target toxicity)
- Design biomarkers for clinical trial patient selection

### Drug Discovery Applications

#### **Repurposing for CNS Disorders**

**Workflow**:
1. Apply RL clustering to drug-treated vs. control scRNA-seq
2. Identify clusters with restored normal GAG profiles
3. Characterize molecular mechanisms
4. Predict optimal dosing/timing

**Example**: Testing chondroitinase ABC for stroke recovery
- scRNA-seq from peri-infarct tissue ± drug
- RL identifies PNN+ interneuron states restored by treatment
- Informs optimal treatment window and dose

#### **Patient Stratification**

**Precision medicine approach**:
- scRNA-seq from patient biopsies (e.g., epilepsy surgery tissue)
- RL clustering reveals patient-specific PNN/sulfation states
- Predict treatment response based on molecular profile
- Select therapy: PNN-targeting vs. traditional anti-epileptics

### Broader Scientific Applications

#### **Beyond Brain: ECM in Other Tissues**

Our framework generalizes to any tissue where ECM heterogeneity matters:

**Cancer**:
- Tumor microenvironment: Fibroblast states with distinct ECM production
- Metastasis: GAG-sulfation changes during EMT
- Drug resistance: ECM barriers to drug penetration

**Fibrosis**:
- Cardiac fibrosis: Myofibroblast activation states
- Liver cirrhosis: Stellate cell heterogeneity
- Pulmonary fibrosis: ECM remodeling signatures

**Immune**:
- T cell activation: Glycocalyx changes during differentiation
- Macrophage polarization: M1/M2 states with distinct ECM interactions

**Method**: Replace GAG-sulfation gene sets with tissue-specific ECM programs

#### **Beyond ECM: Any Gene Program**

**Metabolic states**:
- Glycolysis vs. OXPHOS in T cells (exhaustion)
- Lipid metabolism in adipocytes (browning)
- Amino acid metabolism in cancer cells (auxotrophy)

**Stress responses**:
- Heat shock proteins (proteotoxic stress)
- Unfolded protein response (ER stress)
- Hypoxia response (HIF pathway)

**Developmental programs**:
- HOX gene expression (positional identity)
- Lineage TFs (hematopoiesis, neurogenesis)
- Cell cycle phase (G1/S/G2/M)

**Method**: Swap gene sets, re-train or transfer-learn RL policy

### Research Tool Impact

#### **Accelerating Discovery**

**Current bottleneck**: Manual cluster annotation takes days-weeks
- Expert must inspect marker genes, literature, pathways
- Subjective, not reproducible
- Doesn't scale to atlas-level projects (millions of cells)

**Our solution**: Automated, biology-aware refinement in hours
- Specify gene program of interest → RL discovers relevant subtypes
- Reproducible across labs
- Scales to large datasets

**Time savings**: 10-100x faster than manual curation

#### **Enabling Large-Scale Projects**

**Human Cell Atlas**: Comprehensive map of all human cell types
- Needs: Automated, consistent annotation across tissues/labs
- Challenge: Millions of cells, thousands of cell types
- Our contribution: Scalable, knowledge-guided clustering

**Disease cell atlases**: Mapping cellular changes in disease
- Needs: Identify disease-associated states across patients
- Challenge: Heterogeneity (genetic, environmental, technical)
- Our contribution: Robust refinement integrating disease pathways

### Implementation in Clinical Pipelines

#### **Near-term** (1-3 years)

**Research hospitals with genomics cores**:
- Integrate RL clustering into scRNA-seq analysis pipelines
- Custom gene sets for disease of interest (e.g., cancer pathways)
- Complement standard Seurat/Scanpy workflows

**Use case**: Tumor profiling for precision oncology
- scRNA-seq from biopsy → RL identifies drug-resistant clones → guide treatment

#### **Long-term** (5-10 years)

**Clinical diagnostics**:
- FDA-approved assay: scRNA-seq + RL clustering for disease classification
- Reimbursable test: Identifies actionable molecular subtypes
- Point-of-care: Cloud-based analysis platform

**Example**: Epilepsy surgical planning
- scRNA-seq from pre-surgical biopsy
- RL identifies seizure-generating cell states
- Informs extent of resection needed

---

## 9. Realistic Timeline

### Assumption: Using Claude Code for Implementation

**Efficiency gains**:
- Code generation: 50-60% faster
- Debugging: 40% faster (guided troubleshooting)
- Documentation: 70% faster (automated docstrings)
- Testing: 30% faster (test template generation)

**Overall speedup**: **~1.5-2x faster than manual coding**

### Phase-by-Phase Timeline

#### **Phase 0: Setup** (1 day)
**Tasks**:
- Project structure creation
- Conda environment setup
- Dependency installation
- Hardware verification (MPS)

**With Claude Code**:
- Automated: environment.yml generation, directory structure
- Manual: Reviewing output, testing installation
- **Time**: 2-3 hours

---

#### **Phase 1: Data & Gene Sets** (2-3 days)

**Week 1, Days 1-3**

**Tasks**:
- Download dataset (Allen Brain or Mathys et al.)
- Preprocessing pipeline (QC, normalization, scVI)
- Extract inhibitory interneurons
- Curate GAG-sulfation gene sets
- Exploratory analysis notebook

**With Claude Code**:
- Data download/preprocessing: **4 hours** (automated pipeline, manual QC checks)
- Gene set curation: **3 hours** (semi-automated GO/Reactome queries, manual validation)
- Exploration notebook: **2 hours** (plotting templates)

**Deliverables**:
- `data/interneurons_subset.h5ad` (2k-5k cells)
- `gene_sets/gag_sulfation_sets.json` (7 gene sets)
- `notebooks/01_data_exploration.ipynb`

**Time**: **8-10 hours**

---

#### **Phase 2: Reward Functions** (1-2 days)

**Week 1, Days 3-4**

**Tasks**:
- Implement clustering quality metrics (silhouette, modularity, balance)
- Implement GAG enrichment functions (AUCell, ANOVA)
- Implement penalty functions
- Implement composite reward
- Testing notebook

**With Claude Code**:
- Core functions: **3 hours** (guided implementation, automated testing)
- Validation notebook: **2 hours** (test on dummy clusterings)
- Debugging: **1 hour**

**Deliverables**:
- `environment/rewards.py` (all metrics)
- `environment/penalties.py`
- `tests/test_rewards.py`
- `notebooks/02_reward_validation.ipynb`

**Time**: **5-6 hours**

---

#### **Phase 3: Clustering Actions** (1-2 days)

**Week 1, Days 4-5**

**Tasks**:
- Implement split/merge/recluster operations
- Implement helper functions (centroids, distances)
- Unit tests
- Action validation notebook

**With Claude Code**:
- Action functions: **3 hours**
- Helpers + tests: **2 hours**
- Validation notebook: **1.5 hours**

**Deliverables**:
- `environment/actions.py`
- `environment/action_utils.py`
- `tests/test_actions.py`
- `notebooks/03_action_validation.ipynb`

**Time**: **6-7 hours**

---

#### **Phase 4: RL Environment** (2-3 days)

**Week 2, Days 1-3**

**Tasks**:
- Implement state extraction
- Implement Gym environment class
- Integration testing
- Environment validation notebook

**With Claude Code**:
- State extraction: **2 hours**
- Environment class: **4 hours** (debugging episode logic)
- Testing: **2 hours** (check_env, random agent)
- Validation: **1 hour**

**Deliverables**:
- `environment/state.py`
- `environment/clustering_env.py`
- `tests/test_environment.py`
- `notebooks/04_environment_validation.ipynb`

**Time**: **8-10 hours**

---

#### **Phase 5: Baselines** (1 day)

**Week 2, Day 3**

**Tasks**:
- Implement Leiden grid search
- Implement greedy heuristic
- Implement Bayesian optimization
- Implement GAG-only hierarchical
- Baseline comparison notebook

**With Claude Code**:
- All baseline methods: **3 hours** (template-based)
- Comparison script: **1 hour**
- Analysis notebook: **2 hours**

**Deliverables**:
- `baselines/leiden_grid_search.py`
- `baselines/greedy_heuristic.py`
- `baselines/bayesian_opt.py`
- `baselines/hierarchical_gag.py`
- `results/baselines/metrics.csv`
- `notebooks/05_baseline_analysis.ipynb`

**Time**: **5-6 hours**

---

#### **Phase 6: RL Training** (1-2 days)

**Week 2, Days 4-5**

**Tasks**:
- Implement training script
- Implement callbacks (logging, checkpointing)
- Test training run (5k steps)
- Full training run (50k steps)
- Evaluate trained policy

**With Claude Code**:
- Training infrastructure: **3 hours**
- Test run + debugging: **2 hours**
- Full training: **1 hour** (mostly waiting)
- Evaluation script: **2 hours**

**Deliverables**:
- `rl/train_ppo.py`
- `rl/callbacks.py`
- `rl/evaluate.py`
- `runs/exp_001/final_model.zip`
- Training logs (TensorBoard)

**Time**: **7-8 hours** (+ 1 hour compute time)

---

#### **Phase 7: Evaluation & Visualization** (2-3 days)

**Week 3, Days 1-3**

**Tasks**:
- RL inference (5 runs)
- Comparison table (RL vs. baselines)
- All visualizations (UMAPs, heatmaps, metrics)
- Differential expression
- Results analysis notebook
- Generate publication figures

**With Claude Code**:
- Comparison infrastructure: **2 hours**
- Visualization functions: **3 hours** (plotting templates)
- DE analysis: **2 hours**
- Results notebook: **3 hours** (interpretation, writing)
- Figure generation script: **2 hours**

**Deliverables**:
- `evaluation/compare_methods.py`
- `evaluation/visualizations.py`
- `evaluation/differential_expression.py`
- `notebooks/06_results_analysis.ipynb`
- `figures/` (all publication plots)
- `results/comparison_table.csv`

**Time**: **12-15 hours**

---

#### **Phase 8: Documentation & Packaging** (1-2 days)

**Week 3, Days 3-4**

**Tasks**:
- Update README with full instructions
- Write API documentation
- Write user guide
- Create end-to-end pipeline script
- Freeze requirements
- Add example outputs
- Final testing on clean environment

**With Claude Code**:
- Documentation: **3 hours** (automated docstring extraction, template filling)
- Pipeline script: **1 hour**
- Testing/validation: **2 hours**
- Polish: **2 hours**

**Deliverables**:
- `README.md` (comprehensive)
- `docs/API.md`
- `docs/USER_GUIDE.md`
- `scripts/run_full_pipeline.sh`
- `requirements_frozen.txt`
- `results/example_outputs/`
- `LICENSE`, `CHANGELOG.md`

**Time**: **7-8 hours**

---

### Summary Timeline

| Phase | Description | Time (hours) | Calendar Days |
|-------|-------------|--------------|---------------|
| 0 | Setup | 2-3 | 0.5 |
| 1 | Data & Gene Sets | 8-10 | 2-3 |
| 2 | Reward Functions | 5-6 | 1-2 |
| 3 | Clustering Actions | 6-7 | 1-2 |
| 4 | RL Environment | 8-10 | 2-3 |
| 5 | Baselines | 5-6 | 1 |
| 6 | RL Training | 7-8 | 1-2 |
| 7 | Evaluation | 12-15 | 2-3 |
| 8 | Documentation | 7-8 | 1-2 |
| **Total** | **60-73 hours** | **~15-20 working days** |

### Realistic Schedules

#### **Full-Time Focus** (40 hours/week)
- **Duration**: 1.5-2 weeks
- **Schedule**: 6-8 hours/day, 5 days/week
- **Milestones**:
  - End of Week 1: Phases 0-4 complete (environment functional)
  - Mid Week 2: Phases 5-6 complete (RL trained)
  - End Week 2: Phases 7-8 complete (results analyzed, documented)

#### **Part-Time (Evenings/Weekends)** (15-20 hours/week)
- **Duration**: 3-4 weeks
- **Schedule**: 2-3 hours/evening (3-4 days) + 4-6 hours/weekend
- **Milestones**:
  - End of Week 1: Phases 0-2 complete
  - End of Week 2: Phases 3-4 complete
  - End of Week 3: Phases 5-6 complete
  - End of Week 4: Phases 7-8 complete

#### **Sprint Mode** (Intensive)
- **Duration**: 5-7 days
- **Schedule**: 10-12 hours/day
- **Milestones**:
  - Days 1-2: Phases 0-3
  - Days 3-4: Phases 4-5
  - Day 5: Phase 6
  - Days 6-7: Phases 7-8

### Buffer for Unexpected Issues

**Add 20-30% buffer for**:
- Dataset download issues (server downtime, access problems)
- RL training instability (hyperparameter tuning needed)
- Reward hacking (need to redesign penalties)
- Bugs in environment (edge cases, integration issues)
- GAG signal weaker than expected (need alternative gene sets)

**Recommended timeline with buffer**: **2-3 weeks part-time** or **1.5-2 weeks full-time**

### Critical Path Items

**Must complete before next phase**:
1. Phase 1 → Phase 2: Need processed data for testing rewards
2. Phase 2 → Phase 3: Reward functions used to evaluate actions
3. Phase 3 → Phase 4: Actions integrated into environment
4. Phase 4 → Phase 6: Environment must be stable before RL training
5. Phase 6 → Phase 7: Trained model needed for evaluation

**Can parallelize**:
- Phase 2 & 3: Rewards and actions are independent
- Phase 5: Baselines can run alongside Phase 6 training
- Phase 8: Documentation can start during Phase 7

### Daily Work Plan Example (Full-Time)

**Week 1**
- **Monday**: Phase 0 (setup) + start Phase 1 (data download)
- **Tuesday**: Finish Phase 1 (preprocessing, gene sets, exploration)
- **Wednesday**: Phase 2 (reward functions) + Phase 3 (actions) start
- **Thursday**: Finish Phase 3, start Phase 4 (environment implementation)
- **Friday**: Finish Phase 4, validate with random agent

**Week 2**
- **Monday**: Phase 5 (baselines)
- **Tuesday**: Phase 6 (RL training setup, test run)
- **Wednesday**: Phase 6 continued (full training, evaluation)
- **Thursday**: Phase 7 (evaluation, visualization, DE analysis)
- **Friday**: Phase 7 continued (results interpretation, figures)

**Week 3** (if needed)
- **Monday-Tuesday**: Phase 8 (documentation, packaging)
- **Wednesday**: Buffer for any incomplete items
- **Thursday**: Final testing and polish
- **Friday**: Submission/presentation prep

---

## 10. Ethical, Practical, and Risk Considerations

### Ethical Considerations

#### **Data Privacy and Consent**

**Issue**: scRNA-seq data may contain identifiable genetic information

**Our approach**:
- ✅ Use **publicly available datasets** with appropriate data use agreements
- ✅ Datasets chosen have **consent for secondary research use**
- ✅ No individual-level genetic information reported (only aggregated cell type results)
- ✅ Follow data provider's terms of use (citation, acknowledgment)

**Specific datasets**:
- Allen Brain Cell Types: Public domain, no restrictions
- Mathys et al. (2019): Synapse platform, open access tier
- All donors provided informed consent for data sharing

**Protection measures**:
- No re-identification attempts
- No linkage to clinical phenotypes beyond published metadata
- No sharing of raw data (only processed, aggregated results)

#### **Algorithmic Bias**

**Issue**: RL agent could learn biased strategies if training data is non-representative

**Our approach**:
- ✅ Use **diverse donor samples** (multiple individuals, demographics if available)
- ✅ **Validate across datasets** (test generalization, not just training performance)
- ✅ **Report failure modes** (when does RL not work?)
- ✅ Avoid over-claiming (frame as "exploration" not "ground truth")

**Specific mitigation**:
- If dataset lacks diversity (e.g., all male donors), explicitly state limitation
- Test on independent validation cohorts
- Report performance stratified by batch/donor if applicable

#### **Misuse Potential**

**Issue**: Could RL framework be misused for identifying vulnerable populations?

**Risk assessment**: **LOW**
- Our method improves clustering, not surveillance
- Requires specialized expertise (not accessible to bad actors)
- Single-cell data is expensive, not widely available

**Safeguards**:
- Open-source code includes ethical use guidelines
- No examples of using RL for discriminatory clustering
- Focus on biomedical discovery, not human classification

#### **Clinical Translation Concerns**

**Issue**: Premature clinical use before validation

**Our approach**:
- ✅ Clear statement: **"Research use only, not for clinical decisions"**
- ✅ Require extensive validation before clinical application
- ✅ Regulatory pathway (FDA, CE marking) needed for diagnostics
- ✅ Clinical trial data required for treatment guidance

**Timeline expectation**: 5-10 years before clinical use (if ever)

### Practical Considerations

#### **Computational Accessibility**

**Challenge**: Not everyone has access to M5 Mac or GPUs

**Solutions**:
- ✅ **CPU fallback**: Code works on CPU (slower but functional)
- ✅ **Cloud options**: Google Colab, AWS, Azure notebooks
- ✅ **Pre-trained models**: Share trained policies for inference-only use
- ✅ **Lightweight version**: Reduced action space for faster training

**Documentation**:
- Clearly state hardware requirements
- Provide cloud deployment instructions
- Estimate costs for cloud compute

#### **Scalability**

**Current scope**: 2k-5k cells (single cell type)

**Scaling challenges**:
| Challenge | Issue | Solution |
|-----------|-------|----------|
| **Large datasets** | 100k+ cells slow down reward computation | Subsample for silhouette, approximate metrics |
| **Many cell types** | State space grows with cluster count | Hierarchical approach (coarse → fine) |
| **Memory** | Full dataset in RAM | Streaming, or split by major type |
| **Training time** | More complex state/action → longer training | Transfer learning, multi-task RL |

**Future work**:
- Optimize for atlas-scale datasets (millions of cells)
- Distributed RL training (multiple GPUs)
- Online learning (update policy as new data arrives)

#### **Interpretability**

**Challenge**: RL policy is a "black box" neural network

**Approaches to interpretability**:
1. **Action logging**: Record which actions chosen in which states
2. **Saliency maps**: Which state features drive action selection?
3. **Policy distillation**: Approximate RL policy with decision tree
4. **Ablation studies**: Remove state features, measure impact
5. **Qualitative analysis**: Manually inspect episode trajectories

**Deliverable**: Interpretability notebook showing:
- Action frequency by state
- Feature importance for action selection
- Example successful vs. failed episodes

#### **Maintenance and Support**

**Long-term sustainability**:
- ✅ **GitHub repository**: Issues, pull requests, community contributions
- ✅ **Documentation website**: Hosted on Read the Docs or GitHub Pages
- ✅ **Versioning**: Semantic versioning (v1.0.0, v1.1.0, etc.)
- ✅ **CI/CD**: Automated testing on commits (GitHub Actions)

**Community engagement**:
- Respond to issues within 1 week
- Accept pull requests (with code review)
- Annual dependency updates (Scanpy, PyTorch versions)

**Longevity plan**:
- If project gains traction: Form working group, multi-lab collaboration
- If limited interest: Archive with clear "maintenance mode" status
- Succession plan: Train junior researchers to take over

### Risk Mitigation Strategies

#### **Risk 1: RL Training Fails to Converge**

**Probability**: 10-15%

**Symptoms**:
- Reward does not increase over training
- Agent degenerates to always choosing same action
- High variance in episode rewards

**Mitigation**:
- **Preventive**:
  - Use proven PPO hyperparameters
  - Entropy regularization (ent_coef=0.01)
  - Reward normalization
  - Clip extreme rewards

- **Reactive**:
  - Reduce learning rate (3e-4 → 1e-4)
  - Increase entropy coefficient (0.01 → 0.05)
  - Simplify action space (remove least-used actions)
  - Try A2C instead (simpler algorithm)

**Fallback**: If RL doesn't work, project still valuable:
- Novel environment for community
- Comprehensive baseline comparisons
- Biology-aware reward function design (useful for future work)

#### **Risk 2: Baselines Outperform RL**

**Probability**: 30-40% (this is actually a reasonable outcome!)

**Scenarios**:
- Greedy heuristic achieves similar composite reward
- Bayesian optimization finds optimal hyperparameters efficiently
- GAG signal not complex enough to require multi-step planning

**Mitigation**:
- **Framing**: Emphasize framework contribution, not just performance
- **Analysis**: Deep dive into when/why baselines succeed
- **Extension**: Test on harder datasets (more cell types, weaker signal)

**Spin as positive finding**:
- "We provide rigorous comparison showing when RL is necessary vs. when simpler methods suffice"
- "Our framework enables fair, reproducible benchmarking"
- "Negative results are valuable—guide future method development"

#### **Risk 3: GAG Signal Too Weak**

**Probability**: 15-20%

**Symptoms**:
- GAG gene sets don't separate clusters well
- ANOVA F-statistics are low (<2)
- Reward function dominated by clustering quality, not biology

**Mitigation**:
- **Preventive**:
  - Validate GAG separation in raw data **before** RL training
  - Check that PNN+ markers (ACAN, BCAN) are expressed
  - Ensure sufficient cells (need >100 PNN+ cells)

- **Reactive**:
  - Try alternative gene sets (stress response, metabolism, cell cycle)
  - Focus on different cell type (e.g., astrocytes have strong ECM programs)
  - Adjust β weight (increase biological signal importance)
  - Use continuous reward (correlation) instead of categorical (ANOVA)

**Fallback**: Generalize to "domain-aware clustering framework" with any gene program

#### **Risk 4: Computational Bottleneck**

**Probability**: 20%

**Symptoms**:
- Environment steps take >1 second
- Training projected to take >4 hours
- Memory errors

**Mitigation**:
- **Preventive**:
  - Profile code (identify slow functions)
  - Cache embeddings and graphs
  - Use sparse matrices
  - Subsample for expensive metrics (silhouette on 1000 cells)

- **Reactive**:
  - Reduce cell count (subsample to 2000 cells)
  - Simplify reward (remove slowest component)
  - Reduce parallel environments (4 → 2)
  - Upgrade hardware (cloud GPU)

**Acceptable outcome**: Even if slower, 2-3 hour training is reasonable for research

#### **Risk 5: Reproducibility Issues**

**Probability**: 25%

**Symptoms**:
- Different runs produce very different results
- Others can't reproduce findings
- Package version conflicts

**Mitigation**:
- **Preventive**:
  - Set all random seeds (numpy, torch, scanpy)
  - Freeze requirements (exact versions)
  - Docker container for ultimate reproducibility
  - Extensive documentation

- **Reactive**:
  - Increase training runs (5 → 10), report distributions
  - Provide pre-trained models
  - Extensive troubleshooting guide
  - Video walkthrough of setup

**Gold standard**: Independent lab can reproduce results within 10% margin

### Data Reproducibility and Sharing Plans

#### **Code Availability**

**Repository**: GitHub (public)
- URL: `github.com/[username]/rl-scrna-clustering`
- License: **MIT** (permissive, encourages reuse)
- Contents:
  - All source code
  - Environment definitions (conda, requirements)
  - Unit tests
  - Documentation
  - Example notebooks

**Release strategy**:
- **v0.1**: Initial implementation (during development)
- **v1.0**: Upon project completion (tested, documented)
- **v1.x**: Bug fixes and minor improvements
- **v2.0**: Major extensions (if project continues)

**Archival**: Zenodo DOI for permanent record

#### **Data Availability**

**Primary datasets** (already public):
- Allen Brain Cell Types: Download instructions in README
- Mathys et al.: Synapse ID provided, download script included

**Processed data**:
- `interneurons_subset.h5ad`: Share on Zenodo (< 100 MB)
- scVI latents: Included in .h5ad
- Gene sets: In GitHub repo (JSON file)

**Results data**:
- Trained RL models: Zenodo (too large for GitHub)
- Baseline results: CSV files in GitHub
- Figures: Both PNG (GitHub) and PDF (Zenodo)

**Metadata**:
- Processing scripts with version numbers
- Random seeds used
- Hardware specifications
- Timestamps

#### **Analysis Reproducibility**

**Notebooks**: All analysis notebooks in `notebooks/` directory
- Include markdown explanations
- Show expected outputs
- Runtime estimates

**Pipeline script**: `scripts/run_full_pipeline.sh`
- End-to-end reproduction
- Estimated runtime: 2-3 hours
- Outputs match published results (within stochastic variation)

**Testing**:
- Unit tests for all functions (`pytest tests/`)
- Integration test (full pipeline on toy data)
- CI/CD: GitHub Actions runs tests on every commit

#### **Documentation**

**Multi-level documentation**:

1. **README.md**: Quick start, installation, basic usage
2. **docs/API.md**: Function-by-function reference
3. **docs/USER_GUIDE.md**: Tutorials, examples, FAQs
4. **Docstrings**: Every function has numpy-style docstring
5. **Notebooks**: Extensively commented
6. **Video tutorial** (optional): 10-minute walkthrough on YouTube

**Community support**:
- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Q&A, usage help
- Email: Contact information in README
- Slack/Discord (if community grows)

#### **Preprint and Publication**

**Preprint**: bioRxiv upon project completion
- Full methods
- All results
- Links to code/data
- Preprint DOI for citation

**Peer review**: Submit to journal (see Section 7)
- Revisions will update GitHub repo
- Final published version on journal website
- Author accepted manuscript (AAM) on institutional repository (if required)

**Post-publication**:
- Respond to community feedback
- Bug fixes and improvements
- Potential follow-up studies

---

## Summary and Significance

This project represents a **convergence of cutting-edge machine learning and fundamental single-cell biology**. By developing the first RL environment for scRNA-seq clustering and demonstrating its application to GAG-sulfation biology, we address critical gaps in both fields:

**For machine learning**: Novel application domain, biology-aware reward shaping, interpretable scientific discovery

**For genomics**: Automated, reproducible, biology-guided cluster refinement; reusable framework

**For neuroscience**: Better characterization of PNN-associated cell states with translational relevance

The project is **feasible** (proven technologies, manageable scope, clear fallbacks), **impactful** (open-source tools, methodological innovation, biological insights), and **ethical** (public data, transparent methods, responsible translation).

**Timeline**: 2-3 weeks with Claude Code
**Outcome**: High-quality portfolio project + potential publication + community resource

---

## References and Resources

**Key Papers**:
- Schulman et al. (2017) - Proximal Policy Optimization (PPO)
- Lopez et al. (2018) - scVI for single-cell analysis
- Traag et al. (2019) - Leiden algorithm
- Fawcett et al. (2019) - The extracellular matrix and perineuronal nets in memory (PNN review)

**Software**:
- Gymnasium: https://gymnasium.farama.org
- Stable-Baselines3: https://stable-baselines3.readthedocs.io
- Scanpy: https://scanpy.readthedocs.io
- scVI-tools: https://scvi-tools.org

**Datasets**:
- Allen Brain Cell Types: https://celltypes.brain-map.org
- Mathys et al. (2019): https://www.synapse.org/#!Synapse:syn18485175

**Contact**: [michael.a.haidar@vanderbilt.edu | Michael Haidar]

---

*This proposal provides a comprehensive roadmap for developing and validating an RL-based framework for biology-aware scRNA-seq clustering, with clear milestones, risk mitigation strategies, and plans for reproducible, ethical research.*
```
