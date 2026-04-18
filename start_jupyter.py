
# USE `modal run start_jupyter.py` to run
# Note, the printed link is broken at the linebreak

# TO sync the dataset folder, 
# `modal volume rm -r datasets-vol /datasets`
# `modal volume put datasets-vol .\datasets /`

# TO sync local code for remote imports:
'''
modal volume rm -r code-vol /surrogate_sv
modal volume put code-vol .\surrogate_sv /surrogate_sv
'''

import modal
import secrets
from pathlib import Path

# 1. Define your "Dream Environment"
requirements_path = Path(__file__).with_name("requirements.txt")

# Looks for .env starting from this script's directory
dotenv_secret = modal.Secret.from_dotenv(__file__)

datasets_vol = modal.Volume.from_name("datasets-vol", create_if_missing=True)

# This will store your models and SAE weights permanently
models_vol = modal.Volume.from_name("models-cache-vol", create_if_missing=True)
code_vol = modal.Volume.from_name("code-vol", create_if_missing=True)



my_image = (
    modal.Image.debian_slim()
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
    )
    .pip_install(
        "uv",
        "jupyterlab",
        "transformer_lens",
        "sae_lens", # Added since you use it in imports
        "accelerate",
        "huggingface_hub",
        "matplotlib",
        "circuitsvis",
        "plotly",
        "jaxtyping",
        "docent-python"
    )
)

app = modal.App("custom-mech-interp", image=my_image)

@app.local_entrypoint()
def main():
    print("🚀 Spinning up your custom A100 environment...")

    token = secrets.token_urlsafe(16)

    # 2. Launch the sandbox
    # FIX: We explicitly pass 'app=app' so it knows which session to attach to
    sandbox = modal.Sandbox.create(
        "jupyter", "lab",
        "--no-browser",
        "--ip=0.0.0.0",
        "--port=8888",
        "--allow-root",
        "--ServerApp.root_dir=/workspace",
        f"--ServerApp.token={token}",
        "--ServerApp.disable_check_xsrf=True",
        "--ServerApp.allow_origin='*'",
        image=my_image,
        volumes={
            "/root/datasets": datasets_vol,
            "/root/cache": models_vol,
            "/workspace": code_vol,
        },
        gpu="A100-80GB",  # CHANGE THIS TO PREFERRED to "T4" if you want to save credits
        encrypted_ports=[8888],
        timeout=7200, # Safety net: auto-shutdown after 2 hour
        app=app,
        secrets=[
            dotenv_secret,
            modal.Secret.from_dict({"HF_HOME": "/root/cache", "PYTHONPATH": "/workspace"})
            ],  # injects .env keys as env vars
    )

    # 3. Get the secure URL
    tunnel_url = sandbox.tunnels()[8888].url
    print(f"\n✅ SUCCESS! Your GPU Server is ready.")
    print(f"🔗 COPY THIS URL into VS Code 'Existing Jupyter Server':")
    print(f"{tunnel_url}/lab?token={token}\n")
    print("📦 Code path mounted at /workspace (from Modal volume: code-vol)")
    
    # 4. Keep it running until you stop it
    try:
        sandbox.wait()
    except KeyboardInterrupt:
        print("Stopping server...")
        sandbox.terminate()