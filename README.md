
# AI Weight Visualizer

An interactive 3D sphere made from transformer weights. Each sector represents a transformer block: positive weights push outward, negative weights pull inward, and zero sits on the reference sphere.

Built as a small experiment: **what would a model look like if you could see its weights as a physical shape?**

## Features

- Whole-model view with one sector per transformer block.
- Layer slider to expand a single block while keeping the camera angle fixed.
- Drag to rotate, scroll to zoom, and adjust radial relief.
- Hover over points to inspect tensor names, coordinates, and weight values.
- Deterministic sampling with matrix minima and maxima preserved.
- Self-contained HTML output that opens locally in a browser.

## Try it

Open **`weight-sphere.html`** in Firefox or another modern browser. The included example contains sampled weights from a small JEV transformer.

No installation, server, GPU, or internet connection is needed to view the generated HTML. JavaScript must be enabled.

## Generate your own view

Requires Python 3.9+ and NumPy. Keep `visualize_weights.py` and `viewer-template.html` together.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python visualize_weights.py "/path/to/best.pt" -o my-model.html
```

Open `my-model.html` in your browser. For a denser view:

```bash
python visualize_weights.py "/path/to/best.pt" \
  -o denser.html --points-per-layer 20000
```

The default budget is approximately 6,000 points per block, plus preserved extrema. Denser views take more browser processing time.

## Model support

**The current generator supports JEV checkpoints, not arbitrary transformer models.** It reads the block count and width automatically, so different-sized JEV models work without changing the viewer.

Expected checkpoint structure:

- `config`: model configuration, including `layers` and `width`.
- `model`: state dictionary containing `blocks.N.attention.qkv.weight`, `blocks.N.attention.output.weight`, `blocks.N.mlp.0.weight`, and `blocks.N.mlp.2.weight`.
- Combined Q/K/V projection and a feed-forward expansion of 4 × model width.

When installed, PyTorch loads checkpoints on the CPU with `weights_only=True`. Otherwise, a restricted NumPy-only loader supports the float32 PyTorch ZIP format used by this project.

## Reading the sphere

Within each sector, matrix rows and columns determine position. Q, K, V, attention output, and the two feed-forward matrices occupy bands proportional to their parameter counts.

All blocks share one linear weight scale within a view. Colors identify blocks; relief controls displacement. Embeddings, biases, and normalization parameters are excluded from the sphere.

**This is a visualization of parameter values, not neural connectivity or live activity.** Nearby points are neighbors in the chosen layout. Spikes do not directly measure importance, specialization, or model quality. Initialization can produce visible differences between matrix types.

Scales are calculated independently for each checkpoint, so separate views are not automatically comparable. Generated HTML embeds the displayed weight values.

## Possible next steps

Adapters for other architectures, checkpoint comparisons with shared scaling, weight-based colors, onion-style layer views, and Blender export.

Ideas, issues, and contributions are welcome.
