#!/bin/bash
# Conda environment setup for Isaac Sim 4.x on Blackwell GPUs (RTX 5090/5050/etc.)
#
# Usage:
#   export ISAACSIM_PATH=/path/to/isaac-sim
#   bash setup_new.sh

set -e

ENV_NAME="NavRL"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ORBIT_PATH="${SCRIPT_DIR}/third_party/orbit"

# --- Validate ---
if [ -z "${ISAACSIM_PATH}" ]; then
    echo "[ERROR] ISAACSIM_PATH is not set."
    echo "  export ISAACSIM_PATH=/path/to/isaac-sim"
    exit 1
fi
echo "[INFO] Isaac Sim: ${ISAACSIM_PATH}"

eval "$(conda shell.bash hook)"

# Accept conda TOS (required on fresh installs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# --- Step 1: Create conda env ---
if conda env list | grep -qw "${ENV_NAME}"; then
    echo "[INFO] Env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[INFO] Creating conda env '${ENV_NAME}' with python=3.10..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

# --- Step 2: Symlink orbit to Isaac Sim ---
echo "[INFO] Creating _isaac_sim symlink..."
cd "${ORBIT_PATH}"
[ -L "_isaac_sim" ] && rm -f "_isaac_sim"
ln -s "${ISAACSIM_PATH}" "_isaac_sim"

# --- Step 3: Conda activate/deactivate hooks ---
echo "[INFO] Setting up conda activation hooks..."
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/env_vars.sh" << EOF
#!/usr/bin/env bash
echo "Setup Isaac Sim Conda environment."
export PYTHONPATH_PREV=\$PYTHONPATH
export LD_LIBRARY_PATH_PREV=\$LD_LIBRARY_PATH
source "${ISAACSIM_PATH}/setup_conda_env.sh"
# Prepend conda site-packages so pip-installed torch takes precedence
# over Isaac Sim's bundled torch (2.2.2+cu118, no sm_120 support)
export PYTHONPATH="\${CONDA_PREFIX}/lib/python3.10/site-packages:\${PYTHONPATH}"
export RESOURCE_NAME="IsaacSim"
EOF

cat > "${DEACTIVATE_DIR}/env_vars.sh" << 'EOF'
#!/usr/bin/env bash
unset CARB_APP_PATH EXP_PATH ISAAC_PATH RESOURCE_NAME
export PYTHONPATH="${PYTHONPATH_PREV}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREV}"
EOF

conda install -c conda-forge -y importlib_metadata > /dev/null 2>&1 || true
conda activate "${ENV_NAME}"

# --- Step 4: Rename Isaac Sim's bundled torch (no sm_120 support) ---
# Isaac Sim 4.x bundles torch 2.2.2+cu118 which has no Blackwell GPU kernels.
# Renaming prevents it from shadowing the correct torch we install below.
ML_PREBUNDLE="${ISAACSIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle"
for pkg in torch torchvision torchaudio; do
    if [ -d "${ML_PREBUNDLE}/${pkg}" ] && [ ! -d "${ML_PREBUNDLE}/${pkg}_bak" ]; then
        echo "[INFO] Renaming bundled ${pkg} → ${pkg}_bak..."
        mv "${ML_PREBUNDLE}/${pkg}" "${ML_PREBUNDLE}/${pkg}_bak"
    fi
done

# --- Step 5: Install PyTorch with CUDA 12.8 (sm_120 / Blackwell support) ---
echo "[INFO] Installing PyTorch 2.x+cu128 (Blackwell compatible)..."
PYTHONPATH="" "${CONDA_PREFIX}/bin/pip" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# --- Step 6: Pip packages ---
echo "[INFO] Installing pip packages..."
PYTHONPATH="" "${CONDA_PREFIX}/bin/pip" install numpy==1.26.4
PYTHONPATH="" "${CONDA_PREFIX}/bin/pip" install \
    imageio-ffmpeg==0.4.9 "moviepy==1.0.3" \
    "hydra-core>=1.3" omegaconf \
    einops pyyaml rospkg matplotlib tomli wandb \
    prettytable==3.3.0 hidapi "gymnasium==0.29.0" trimesh "pyglet<2" toml

# --- Step 7: Orbit extensions ---
echo "[INFO] Installing orbit extensions..."
find -L "${ORBIT_PATH}/source/extensions" -mindepth 1 -maxdepth 1 -type d | while read ext; do
    if [ -f "${ext}/setup.py" ]; then
        echo "  -> ${ext}"
        PYTHONPATH="" "${CONDA_PREFIX}/bin/pip" install --no-deps --no-build-isolation -e "${ext}"
    fi
done

# --- Step 8: OmniDrones ---
echo "[INFO] Installing OmniDrones..."
cp -r "${SCRIPT_DIR}/third_party/OmniDrones/conda_setup/etc" "${CONDA_PREFIX}"
conda activate "${ENV_NAME}"
PYTHONPATH="" "${CONDA_PREFIX}/bin/pip" install -e "${SCRIPT_DIR}/third_party/OmniDrones"

# --- Step 9: Rebuild tensordict against new torch ---
echo "[INFO] Installing tensordict..."
cd "${SCRIPT_DIR}/third_party/tensordict"
PYTHONPATH="" "${CONDA_PREFIX}/bin/python" setup.py develop

# --- Step 10: Rebuild TorchRL against new torch ---
echo "[INFO] Installing TorchRL..."
cd "${SCRIPT_DIR}/third_party/rl"
PYTHONPATH="" "${CONDA_PREFIX}/bin/python" setup.py develop

# --- Verify ---
echo ""
conda activate "${ENV_NAME}"
python -c "
import torch
print('[INFO] torch:', torch.__version__)
print('[INFO] CUDA:', torch.version.cuda)
print('[INFO] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('[INFO] sm:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')
x = torch.zeros(4, device='cuda')
print('[INFO] CUDA tensor OK:', x)
"

echo ""
echo "=========================================="
echo " Setup complete!  conda activate ${ENV_NAME}"
echo "=========================================="
echo ""
echo "[NOTE] Isaac Sim 4.x uses 'omni.isaac.lab' (Isaac Lab) instead of"
echo "       'omni.isaac.orbit'. Training scripts import 'isaacsim' before"
echo "       'omni.isaac.kit' — this is already handled in the training scripts."
