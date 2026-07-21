# 01 - Planning and Brainstorming Notes

## Project Context
The primary objective of this project is to automate the functional Region of Interest (fROI) delineation process on the cortical surface of the brain, specifically targeting the Visual Word Form Area (VWFA) subregions (anterior, central, and posterior: aVWFA, cVWFA, pVWFA). 

Expert manual delineation of fROIs is a painstaking, subjective process. The expert relies on visual "gradation" (the spatial gradient and local contrast of activation) to draw outline boundaries. 

## Cognitive Pitfalls addressed
1. **Anchoring & Confirmation Bias**: In our initial brainstorming, we recognized the temptation to anchor on the previously implemented "local binning and scaling" method. This previous method divided the brain into 10 sliding bins along the anterior-posterior (MNI-Y) axis and scaled activations within each bin to force weak anterior activations to appear stronger. While this "manipulation of activations" achieved the goal of making anterior ROIs visible to a hard-thresholded BFS search, it was unscientific, introduced spatial artifacts, and violated physiological principles by scaling noise in inactive regions.
2. **First-Principles Re-evaluation**: By stepping back and analyzing the expert's visual strategy, we realized that an expert does not artificially amplify weak signals. Instead, they scale their boundary-drawing criteria relative to the local peak of activation. This realization led directly to our proposed **Gradation-Constrained Region Growing (GCRG)** algorithm.

## Brainstorming the Perfect Gradation Clustering Algorithm
An elegant, neurologically and mathematically sound algorithm must satisfy three main criteria:
1. **Contiguity Guarantee**: The resulting fROI must be a single contiguous patch of vertices on the cortical surface mesh. This is naturally solved by using a Breadth-First Search (BFS) or Dijkstra-based region growing on the surface topology.
2. **Adaptive Local Thresholding (Relative-Drop)**: 
   - Instead of a global hard threshold (e.g., $t > 1.65$) which completely misses weak activations in the anterior region and spills over in the posterior region, we define a dynamic threshold for each cluster grown from a local peak (seed).
   - Let the seed vertex $s$ have raw activation $T_{\text{seed}}$. The growth cutoff for any neighbor $v$ is:
     $$T_{\text{cutoff}} = \max(T_{\text{min}}, \gamma \cdot T_{\text{seed}})$$
     where $\gamma \in (0, 1)$ is the relative-drop parameter, and $T_{\text{min}}$ is an absolute minimum t-value to prevent leaking into negative or background regions.
3. **Sequential Peak Isolation (Multi-Session Anchored DICE)**:
   - In each round of the algorithm, we find the highest available raw t-value across sessions as the seed.
   - We grow a "native" cluster from this seed.
   - We then "anchor" this cluster in all other sessions by finding the peak in the intersection of the native cluster and the other sessions' available vertices, and growing a corresponding cluster there.
   - We compute the cross-session consensus (mean pairwise DICE coefficient). If the consensus is high (DICE > 0), this represents a robust functional ROI, and we claim its union vertices, excluding them from subsequent rounds.
   - This sequential search naturally finds the strongest peaks first (posterior,central) and then proceeds to the weaker peaks (anterior) without needing any artificial local scaling!

## Selected Parameters (Grid-Validated)
- `relative_drop` ($\gamma$): **0.35** (35% drop from the local peak). This value was found to provide the most stable, biologically plausible fROI sizes across subjects.
- `absolute_min` ($T_{\text{min}}$): **1.0**. This allows the weak anterior activation humps to be fully captured even when their boundary drops below the traditional statistical threshold of 1.65, matching the expert's visual strategy.
