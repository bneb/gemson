import os
import time
import subprocess
import runpod
from dotenv import load_dotenv

load_dotenv("../.env")

runpod.api_key = os.getenv("RUNPOD_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

if not runpod.api_key:
    raise ValueError("RUNPOD_API_KEY not found in .env")

ssh_key_path = "/Users/kevin/.ssh/gemson_runpod_ed25519"
if not os.path.exists(ssh_key_path):
    print("Generating temporary SSH key...")
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", ssh_key_path, "-N", ""], check=True)

with open(f"{ssh_key_path}.pub", "r") as f:
    public_key = f.read().strip()

print("Provisioning RTX A5000 on RunPod...")
pod = runpod.create_pod(
    name="Gemson-12B-Trainer",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    gpu_type_id="NVIDIA A100-SXM4-80GB",
    cloud_type="SECURE",
    container_disk_in_gb=100,
    ports="22/tcp",
    env={
        "PUBLIC_KEY": public_key,
        "HF_TOKEN": hf_token
    }
)

pod_id = pod["id"]
print(f"Pod {pod_id} created. Waiting for it to become RUNNING...")

try:
    while True:
        try:
            pod_info = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"RunPod API hiccup: {e}")
            time.sleep(10)
            continue
        if pod_info and pod_info.get("desiredStatus") == "RUNNING" and pod_info.get("runtime"):
            ports = pod_info["runtime"].get("ports", [])
            ssh_port_mapping = next((p for p in ports if p["privatePort"] == 22), None)
            if ssh_port_mapping and ssh_port_mapping.get("ip") and ssh_port_mapping.get("publicPort"):
                break
        print("Sleeping 10s...")
        time.sleep(10)

    ip = ssh_port_mapping["ip"]
    port = ssh_port_mapping["publicPort"]
    print(f"Pod is ready! SSH Address: {ip}:{port}")

    def run_ssh(cmd):
        full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "LogLevel=ERROR", "-i", ssh_key_path, "-p", str(port), f"root@{ip}", cmd]
        print(f"Executing: {cmd}")
        subprocess.run(full_cmd, check=True)

    def run_scp_upload(local_path, remote_path):
        full_cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-i", ssh_key_path, "-P", str(port), local_path, f"root@{ip}:{remote_path}"]
        print(f"Uploading {local_path}...")
        subprocess.run(full_cmd, check=True)

    print("Waiting for SSH daemon to start inside the container...")
    for _ in range(30):
        try:
            run_ssh("echo 'SSH is up!'")
            break
        except subprocess.CalledProcessError:
            print("SSH not ready yet, waiting 5s...")
            time.sleep(5)
    else:
        raise Exception("SSH failed to start after 150 seconds.")

    def run_scp_download(remote_path, local_path):
        full_cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-i", ssh_key_path, "-P", str(port), f"root@{ip}:{remote_path}", local_path]
        print(f"Downloading {remote_path}...")
        for attempt in range(5):
            try:
                subprocess.run(full_cmd, check=True)
                break
            except subprocess.CalledProcessError:
                print(f"Download failed on attempt {attempt+1}. Retrying in 10 seconds...")
                time.sleep(10)
        else:
            raise Exception("Download failed after 5 attempts.")

    run_scp_upload("../data/training_data.jsonl", "/workspace/training_data.jsonl")
    run_scp_upload("train_unsloth.py", "/workspace/train_unsloth.py")
    run_ssh("pip install --upgrade torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124")
    run_ssh("pip uninstall -y torchvision torchaudio torchao")
    run_ssh("pip install git+https://github.com/huggingface/transformers.git")
    run_ssh('pip install peft trl bitsandbytes accelerate huggingface_hub pydantic datasets sentencepiece protobuf xformers cut_cross_entropy "torch==2.6.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124')
    run_ssh("pip install git+https://github.com/unslothai/unsloth-zoo.git --no-deps")
    run_ssh('pip install git+https://github.com/unslothai/unsloth.git --no-deps')
    run_ssh('apt-get update && apt-get install libcurl4-openssl-dev cmake libssl-dev -y')

    print("Executing training pipeline... This will take ~30 minutes.")
    run_ssh("python /workspace/train_unsloth.py")

    os.makedirs("../outputs", exist_ok=True)
    run_scp_download("/workspace/gemson_model_gguf/gemma-4-12b-it.Q4_K_M.gguf", "../outputs/gemson-12b-lora.gguf")
    run_scp_download("/workspace/gemson_model_gguf/gemma-4-12b-it.BF16-mmproj.gguf", "../outputs/gemson-12b-lora-mmproj.gguf")
    print("Download complete!")

finally:
    print(f"Terminating Pod {pod_id} to prevent further billing...")
    runpod.terminate_pod(pod_id)
    print("Pod terminated. Workflow complete.")
