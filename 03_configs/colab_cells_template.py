# Copy these blocks into a Colab notebook as separate cells.

# CELL 1
from google.colab import drive
drive.mount('/content/drive')


# CELL 2
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/mihai-lora-v2-colab')
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'checkpoints'
SAMPLES_DIR = ROOT / 'samples'
LOG_DIR = ROOT / 'logs'

TRIGGER = 'mihai'
TOTAL_STEPS = 1600
CHUNK_SIZE = 400
SAVE_EVERY = 100
SAMPLE_EVERY = 100
LR = 1.5e-4
RANK = 16
RESOLUTION = 1024
BATCH_SIZE = 1

for p in [DATA_DIR, OUT_DIR, SAMPLES_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print('ROOT', ROOT)


# CELL 3 (optional unzip)
import zipfile

SRC_ZIP = '/content/drive/MyDrive/replicate_bundle_v2.zip'
if Path(SRC_ZIP).exists():
    with zipfile.ZipFile(SRC_ZIP, 'r') as zf:
        zf.extractall(DATA_DIR)
    print('Extracted dataset zip.')
else:
    print('Dataset zip missing. Copy files manually into data/.')


# CELL 4
%cd /content
!git clone https://github.com/ostris/ai-toolkit.git
%cd /content/ai-toolkit
!pip -q install -r requirements.txt
!pip -q install accelerate bitsandbytes transformers diffusers safetensors


# CELL 5
import re

def latest_ckpt(path: Path):
    if not path.exists():
        return None, 0
    cands = []
    for p in path.glob('**/*'):
        if p.is_file() and p.suffix in {'.safetensors', '.pt', '.bin'}:
            m = re.search(r'(\d+)', p.stem)
            step = int(m.group(1)) if m else -1
            cands.append((step, p))
    if not cands:
        return None, 0
    step, p = sorted(cands, key=lambda x: x[0])[-1]
    return str(p), max(step, 0)

resume_path, done_steps = latest_ckpt(OUT_DIR)
print('resume_path', resume_path)
print('done_steps', done_steps)


# CELL 6
start_step = done_steps
end_step = min(done_steps + CHUNK_SIZE, TOTAL_STEPS)
print(f'Chunk {start_step} -> {end_step}')


# CELL 7
cfg_text = f"""
job: extension
config:
  name: mihai_lora_v2
  process:
    - type: sd_trainer
      training_folder: "{ROOT}/runs"
      device: cuda:0
      network:
        type: lora
        linear: {RANK}
        linear_alpha: {RANK}
      save:
        dtype: float16
        save_every: {SAVE_EVERY}
        max_step_saves_to_keep: 20
      datasets:
        - folder_path: "{DATA_DIR}"
          caption_ext: "txt"
          default_caption: "photo of {TRIGGER}"
          resolution: [{RESOLUTION}, {RESOLUTION}]
      train:
        batch_size: {BATCH_SIZE}
        steps: {end_step}
        lr: {LR}
        gradient_accumulation_steps: 4
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
"""

cfg_path = ROOT / 'train_chunk.yaml'
cfg_path.write_text(cfg_text)
print('Wrote', cfg_path)


# CELL 8
%cd /content/ai-toolkit
resume_arg = f'--resume "{resume_path}"' if resume_path else ''
cmd = f'python run.py --config "{ROOT}/train_chunk.yaml" {resume_arg}'
print(cmd)
!{cmd}


# CELL 9
resume_path, done_steps = latest_ckpt(OUT_DIR)
print('latest', resume_path)
print('done_steps', done_steps)
print('target', TOTAL_STEPS)
