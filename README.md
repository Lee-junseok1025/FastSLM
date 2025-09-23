# FastALM
FastALM: Hierarchical Frame Q-Former for Effective Audio Modality Adaptation

# 🚀 FastALM
FastALM is a **lightweight Audio-Language Model (ALM)** designed to efficiently handle **long-form audio inputs**.

## 🌟 Features
- 🔊 **HFQ-Former**: Hierarchically compresses high-frame-rate audio features while preserving context
- ⚡ **3-Stage Training**: Cost-effective and fast training strategy
- 🧠 **LLM Adaptation**: Adapts pre-trained LLMs to the speech modality

## 📦 Installation
```bash
git clone https://github.com/username/FastALM.git
cd FastALM
pip install -r requirements.txt
```

## Smaple Inference
```python
import torch
import torchaudio
from model.FastALM import FastALM

model = FastALM(
    embed_dim=2560,
    speech_dim=1280,
    lora=True,
    lora_r=16,
    lora_a=64,
    stage_tokens=[80,80,80],
    compression=True,
    compression_size=50,
    model_name = 'Qwen/Qwen3-4B',
    encoder_mode='large-v3',
    pre_training=False
).cuda()


wav,sample_rate  = torchaudio.load(wav_path)
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
audio = resampler(wav)

basic_prompt = '<|ASR|><|audio_bos|><|AUDIO|><|audio_eos|>\nTranscribe this audio clip into text.'
prompt = [
    {"role": "user", "content":  basic_prompt},
]
conversation = model.tokenizer.apply_chat_template(prompt,add_generation_prompt=True,tokenize=False)
print(conversation)

token = model.tokenizer(conversation,return_tensors='pt').input_ids.cuda()

model.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model.generate(
            input_ids=token,
            audio=audio
        )

print(output[0])
```
