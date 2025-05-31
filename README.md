# VirtualCathLab – Stent‑deployment utilities for vascular meshes with centerlines
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](#license)
[![made with VTK](https://img.shields.io/badge/made%20with-VTK-398593)](https://vtk.org)
[![python 3.9](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/downloads/release/python-3913/)

Research code for *in‑silico* generation of post-stent geometry using SDF indenting based
surface deformation and associated analysis utilities  
*(last tested on macOS 15.4 / Apple‑silicon, Python 3.9, VTK 9.3, JAX 0.4.30).*

---

## Quick install (Mac with **Apple‑silicon**)

> **micromamba** is recommended for Mac with M-series chip as it is very fast, lightweight, and
> coexists happily with Homebrew and system Python.

### 1. Install **micromamba** on Mac with Apple-silicon

```bash
# Home in your $HOME/.local, no sudo needed
curl -L https://micromamba.snakepit.net/api/micromamba/osx-arm64/latest \
     | tar -xvj bin/micromamba
mkdir -p ~/micromamba
mv bin/micromamba ~/micromamba/
echo 'export PATH="$HOME/micromamba:$PATH"' >> ~/.zshrc   # or ~/.bash_profile
source ~/.zshrc                                          # reload shell
```
(See https://mamba.readthedocs.io/en/latest/installation.html for information about installing on other
platforms.)

### 2. Create the virtualcathlab environment

```
micromamba create -y -n virtualcathlab \
    python=3.9.19 \
    numpy=1.24.4 \
    scipy=1.10.1 \
    vtk=9.3.0 \
    -c conda-forge
```

### 3. Activate + add pip‑only packages

First initialize micromamba in the shell before first use:
```
eval "$(micromamba shell hook --shell zsh)"
```

Next, activate the environment and pip install the rest of the required packages:
```
micromamba activate virtualcathlab
pip install --upgrade pip
pip install "jax[cpu]"==0.4.30
pip install pyqt6==6.7
```

### 4. Verify the environment is correct

```
python installation-test.py
```

Expected output:
```
PyQt6 version: 6.7.0
Qt version   : 6.7.0
VTK version  : 9.3.0
```
If the versions match and no segmentation fault occurs, the environment is
ready.

⸻

## Repository layout

| Path / script                         | Purpose                                                                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| deploy_stent_standalone.py                      | Deploy one stent until target radius (no overshoot).                                                                             |
| deploy_stent_with_intermediates.py                      | Same as above but stores a .vtp mesh at every 0.1 cm radius increment.                                                                             |
| deploy_stent_batch.py                | Read multiple stent specifications from a plaintext file and deploy them sequentially (optionally saving intermediates); final meshes/centre‑lines written once all stents are placed. |
| installation-test.py                 | Simple Qt + VTK sanity check used above.                                                                                         |


⸻

## Usage
Remember to always activate the environment first
```
micromamba activate virtualcathlab
```

Single‑stent deployment:
```
python deploy_stent_standalone.py \
       --mesh input_surface.vtp \
       --cline input_centerline.vtp \
       --start 765 \
       --target-R 0.4 \
       --start-R 0.05 \
       --length 3.0 \
       --out-mesh deployed_surface.vtp
```
Single-stent deployment with partially stented intermediate results (saved under folder ``deployed_surface_intermediates``):
```
python deploy_stent_with_intermediates.py \
       --mesh input_surface.vtp \
       --cline input_centerline.vtp \
       --start 765 \
       --target-R 0.4 \
       --start-R 0.05 \
       --length 3.0 \
       --save-step 0.1 \
       --out-mesh deployed_surface.vtp
```

Batch deployment:

Create a text file, for instance, put the following content into a file named ``example_batch_stents.txt``
```
# start_id  length(cm)  target_R(cm)
1000        2.0         0.80
 850        3.0         0.60
 663        1.5         0.50
```
Then run the following
```
python deploy_stent_batch.py \
    --mesh   input_surface.vtp \
    --cline  input_centerline.vtp \
    --batch example_batch_stents.txt # each line is start_id,length,target_R \
    --out-mesh deployed_surface.vtp
```
Note: results from stenting only the prefix subset of the input batch are also saved under the folder ``deployed_surface_prefix_subsets``.

⸻

### Algorithmic highlights
•	A signed‑distance‑function (SDF) composed of the smooth union of many capsules approximates the expanding
stent. Contact enforcement is done via a truncated radial kernel. <br>
•	In the interactive tool, Kelvinlets (smooth fundamental solutions of linear elasticity) provide a
closed‑form, mesh‑free deformation field. <br>
•	Batch script performs serial (non‑overlapping) deployment: later stents
deform the geometry that already contains earlier ones. <br>

For details, see upcoming publication, working in progress.

⸻

### Contributing

Pull requests are welcome!  Please open an issue to discuss substantial
changes.

⸻

### License

MIT.  See LICENSE.

⸻

### Citation

@misc{virtualcathlab2025,
  author    = {Bohan Jeff Li and Contributors},
  title     = {VirtualCathLab: SDF-based stent deployment utilities},
  year      = {2025},
  howpublished = {\url{https://github.com/your‑org/virtualcathlab}}
}


⸻

### Contact

Questions?  Open an issue or ping
Bohan Jeff Li — bohan1@stanford.edu
