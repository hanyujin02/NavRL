#!/bin/bash

# Exit immediately if a command fails
set -e

ENV_NAME="NavRL"
# Use NAVRL_DIR instead of SCRIPT_DIR to avoid collision: conda activate sources
# Isaac Sim's setup_conda_env.sh which overwrites a bare SCRIPT_DIR variable.
NAVRL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ── Prerequisites check ──────────────────────────────────────────────────────
if [ -z "$ISAACSIM_PATH" ]; then
    echo "[ERROR] ISAACSIM_PATH is not set."
    echo "        Add the following to ~/.bashrc and then re-run this script:"
    echo "          export ISAACSIM_PATH=\"/path/to/isaac_sim-2023.1.0-hotfix.1\""
    exit 1
fi

if [ ! -d "$ISAACSIM_PATH" ]; then
    echo "[ERROR] ISAACSIM_PATH does not exist: $ISAACSIM_PATH"
    echo "        Make sure Isaac Sim 2023.1.0-hotfix.1 is installed at that path."
    exit 1
fi
if [ ! -f "$ISAACSIM_PATH/setup_conda_env.sh" ]; then
    echo "[ERROR] $ISAACSIM_PATH does not look like a valid Isaac Sim installation (missing setup_conda_env.sh)."
    exit 1
fi

echo "[INFO] Using Isaac Sim at: $ISAACSIM_PATH"
echo "[INFO] NavRL root: $NAVRL_DIR"

# ── Conda init ───────────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"

# ── Step 1: Create conda env ─────────────────────────────────────────────────
if conda env list | grep -w "^${ENV_NAME}" > /dev/null 2>&1; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists — skipping creation."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' (python=3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# ── Step 2: Orbit symlink + conda activation hooks ───────────────────────────
echo "[INFO] Setting up Orbit..."
cd "$NAVRL_DIR/third_party/orbit"

# Create symlink to Isaac Sim if missing or pointing to wrong target
if [ -L "_isaac_sim" ]; then
    if [ "$(readlink _isaac_sim)" != "$ISAACSIM_PATH" ]; then
        rm -f _isaac_sim
        ln -s "$ISAACSIM_PATH" _isaac_sim
        echo "[INFO] Updated symlink: _isaac_sim -> $ISAACSIM_PATH"
    else
        echo "[INFO] Symlink _isaac_sim already correct — skipping."
    fi
elif [ -e "_isaac_sim" ]; then
    echo "[ERROR] _isaac_sim exists but is not a symlink. Remove it manually."
    exit 1
else
    ln -s "$ISAACSIM_PATH" _isaac_sim
    echo "[INFO] Created symlink: _isaac_sim -> $ISAACSIM_PATH"
fi

# orbit.sh --conda writes the Isaac Sim activation hooks into the conda env
# (it skips env creation when the env already exists)
./orbit.sh --conda "$ENV_NAME"

# Activate so subsequent pip/python calls land in the right env
conda activate "$ENV_NAME"

# ── Step 3: System packages required by Orbit ────────────────────────────────
# cmake and build-essential are needed to compile Orbit extensions.
# If you have sudo access, uncomment the line below; otherwise install manually.
# sudo apt-get update -qq && sudo apt-get install -y cmake build-essential
echo "[INFO] Assuming cmake and build-essential are already installed."
echo "       If not, run: sudo apt-get install -y cmake build-essential"

# ── Step 4: Pip packages ─────────────────────────────────────────────────────
# Use 'python -m pip' to ensure we target the conda env even if 'pip' is not
# on PATH (the NavRL env ships pip3/pip3.10 but not a bare 'pip' symlink).
echo "[INFO] Installing pip packages..."
python -m pip install numpy==1.26.4
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
python -m pip install "pydantic==1.9.2"
python -m pip install imageio-ffmpeg==0.4.9
python -m pip install moviepy==1.0.3
python -m pip install hydra-core==1.3.3
python -m pip install einops==0.8.2
python -m pip install pyyaml
python -m pip install rospkg==1.6.1
python -m pip install "matplotlib==3.7.1"
python -m pip install tomli

# ── Step 5: Install Orbit extensions ─────────────────────────────────────────
echo "[INFO] Installing Orbit extensions..."
cd "$NAVRL_DIR/third_party/orbit"
./orbit.sh --install

# ── Step 6: Setup OmniDrones ─────────────────────────────────────────────────
echo "[INFO] Setting up OmniDrones..."
cd "$NAVRL_DIR/third_party/OmniDrones"
# Copy conda activation/deactivation hooks that wire up Isaac Sim env vars
cp -r conda_setup/etc "$CONDA_PREFIX"
# Reload env so the new hooks take effect
conda activate "$ENV_NAME"
# Install the OmniDrones Python package in editable mode
python -m pip install -e .

# ── Step 7: Verify Isaac Kit import ──────────────────────────────────────────
echo "[INFO] Verifying omni.isaac.kit..."
python -c "from omni.isaac.kit import SimulationApp; print('[OK] omni.isaac.kit')"

# ── Step 8: Install TensorDict from source ───────────────────────────────────
echo "[INFO] Installing TensorDict..."
python -m pip uninstall -y tensordict 2>/dev/null || true
cd "$NAVRL_DIR/third_party/tensordict"
python setup.py develop

# ── Step 9: Install TorchRL from source ──────────────────────────────────────
echo "[INFO] Installing TorchRL..."
cd "$NAVRL_DIR/third_party/rl"
python setup.py develop

# ── Final verification ────────────────────────────────────────────────────────
echo "[INFO] Final verification..."
python -c "import torch; print('[OK] torch', torch.__version__)"
python -c "import tensordict; print('[OK] tensordict')"
python -c "import torchrl; print('[OK] torchrl')"
python -c "import omni_drones; print('[OK] omni_drones')"

echo ""
echo "Setup completed successfully!"
echo "To activate the environment: conda activate ${ENV_NAME}"
