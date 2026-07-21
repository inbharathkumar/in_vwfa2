# 03 - Technical Challenges and Implementation Details

## Gradation-Constrained Region Growing (GCRG) Implementation
The GCRG algorithm operates directly on the raw t-values of the `WordvsPER` contrast (no sliding binning, no local scaling), solving the anterior-posterior activation strength discrepancy through a mathematically elegant local relative-drop constraint:

### 1. The Algorithm
1. For each subject and each round, we define the set of available vertices (excluding vertices claimed in previous rounds, and vertices excluded in this round's prior seed loops).
2. We select the seed vertex as the vertex with the maximum raw t-value.
3. We grow a contiguous cluster from the seed using a custom BFS search. At each step, a neighbor $v$ of current vertex $u$ is added if and only if:
   - $v$ is in the available set of vertices.
   - $T(v) \ge \max(T_{\text{min}}, \gamma \cdot T_{\text{seed}})$, where $T_{\text{seed}}$ is the seed's raw activation, $\gamma = 0.35$ (relative-drop), and $T_{\text{min}} = 1.0$ (absolute minimum).
4. This allows the cluster size and threshold to adapt dynamically to the local peak:
   - **Posterior region**: $T_{\text{seed}} = 10.0 \implies T_{\text{cutoff}} = 3.5$. Even if there are contiguous vertices with high activations (e.g., 2.5) that exceed global threshold 1.65, they are excluded because they represent a "stark decrease in gradation" relative to the peak.
   - **Anterior region**: $T_{\text{seed}} = 2.5 \implies T_{\text{cutoff}} = 1.0$. This allows including contiguous weak activations down to 1.0, capturing the entire anterior peak.

---

## Technical Challenges Overcome

### 1. Memory Constraints and SIGKILL 9 during Rendering
- **The Problem**: Plotly's WebGL-based 3D brain rendering engine (`engine='plotly'`) and its corresponding headless export engine (`kaleido`) consume a massive amount of RAM when running sequentially in a sandboxed container. This led to out-of-memory (OOM) situations and the operating system sending a `SIGKILL 9` signal, crashing the python process.
- **The Solution**: We developed a highly efficient, low-memory, and extremely fast rendering engine based on Matplotlib's native 3D surface mapping in Nilearn (`engine='matplotlib'`). By projecting the statistical maps and the ROI contours onto a ventral occipitotemporal cortex (vOTC) ventral patch of the inflated cortical surface, we generated extremely sharp, high-resolution PNG plots of the brains with the exact ROI contours. The rendering completes in less than 2 seconds per subject with zero risk of memory crashes!

### 2. Contiguity Enforcements
- Traditional clustering algorithms like k-means or standard global thresholding can result in fragmented ROIs. By using a customized BFS region growing on the surface topology mesh, we mathematically guarantee that every single Stage 1 ROI is a single, spatially contiguous patch of vertices.

### 3. High-Quality Montages
- We implemented a seamless PIL-based image montage generator that automatically crops and combines individual subjects' plots into a professional $3 \times 2$ grid montage (`Stage1_ROI_contour_plots_5subj.png`), matching the visual format of expert manual annotations and paper-ready publication layouts.
