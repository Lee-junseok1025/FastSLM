import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

import numpy as np
from torch import Tensor
import whisper
from einops import rearrange
from typing import Dict, Iterable, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling_whisper import AudioEncoder
from peft import (
    LoraConfig, 
    get_peft_model
)

try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

LANGUAGES = {
    "en": "english",
    "ko": "korean"
}

def set_trainable_parameters(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad
    module._requires_grad = requires_grad


class Compressor(nn.Module):
    def __init__(
            self, 
            embed_dim,
            num_heads,
            num_query,
            n_ctx,
        ):
        super(Compressor, self).__init__()

        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads
        self.n_ctx = n_ctx
        
        self.query = nn.Parameter(torch.randn(1,num_query,embed_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        
        self.q_ln = nn.LayerNorm(embed_dim,eps=1e-5)
        self.kv_ln = nn.LayerNorm(embed_dim,eps=1e-5)
        
        self.kv_proj = nn.Identity()
        self.out_proj = nn.Linear(embed_dim,embed_dim)

        self.register_buffer("q_pos_embeds", self.sinusoids(num_query, embed_dim))
        self.register_buffer("kv_pos_embeds", self.sinusoids(n_ctx, embed_dim))
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.q_ln.bias, 0)
        nn.init.constant_(self.q_ln.weight, 1.0)
        nn.init.constant_(self.kv_ln.bias, 0)
        nn.init.constant_(self.kv_ln.weight, 1.0)
    
    def sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


    def forward(self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.q_ln(self.query.expand(x.shape[0], -1, -1).to(x.device))
        x = self.kv_ln(self.kv_proj(x))

        q = rearrange(q + self.q_pos_embeds,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
        k = rearrange(x + self.kv_pos_embeds,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
        v = rearrange(x,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)

        attn = scaled_dot_product_attention(q,k,v)
        attn = rearrange(attn,'b h l d -> b l (h d)')
        x = self.out_proj(attn)
        return x


class  MHSA(nn.Module):
    def __init__(self, 
        embed_dim,
        num_heads,
    ):
        super(MHSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dims = embed_dim // num_heads

        self.q = nn.Linear(embed_dim,embed_dim,bias=True)
        self.k = nn.Linear(embed_dim,embed_dim,bias=False)
        self.v = nn.Linear(embed_dim,embed_dim,bias=True)

        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=True)
    
    def forward(
        self,
        x,
        xa=None,
        mask=None,
    ):
        
        b, tgt_len, dim = x.size()
        q = self.q(x)
        k = self.k(x if xa is None else xa)
        v = self.v(x if xa is None else xa)
       
        q = rearrange(q,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims) 
        k = rearrange(k,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)
        v = rearrange(v,'b l (h d) -> b h l d',h=self.num_heads,d=self.head_dims)

        attn = scaled_dot_product_attention(q,k,v,is_causal=mask is not None)
        attn = rearrange(attn,'b h l d -> b l (h d)')
       
        out = self.out_proj(attn)
        return out

class Attention(nn.Module):
    def __init__(
        self, 
        embed_dim,
        num_heads,
    ):
    
        super(Attention, self).__init__()
        self.attn = MHSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.cross_attn = MHSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.norm1 = nn.LayerNorm(embed_dim,eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim,eps=1e-5)

    def forward(self, 
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(x=self.norm2(x),xa=xa)            
        return x

class Downsampler(nn.Module):
    def __init__(self,embed_dim: int):
        super(Downsampler,self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.ln_post = nn.LayerNorm(embed_dim,eps=1e-5)

    def forward(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.ln_post(x)
        return x

    def sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class SpeechEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        speech_dim,
        n_audio_ctx=1500,
        compression=True,
        stage_tokens=[80,80,80],
        compression_size=80,
        mel_dim=128,
        depths=2,
        n_layer=32,
        n_ctx=1500
    ):
        super(SpeechEncoder,self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_dim = embed_dim
        self.n_audio_ctx = n_audio_ctx
        self.mel_dim = mel_dim
        num_heads = speech_dim // 64
        self.compression = compression
        self.compression_size = compression_size
        self.whisper = AudioEncoder(
            n_mels=mel_dim,
            n_ctx=n_ctx,
            n_state=speech_dim,
            n_head=num_heads,
            n_layer=n_layer
        )
        self.whisper.eval()
        set_trainable_parameters(self.whisper,False)
        self.llm_proj = nn.Linear(speech_dim, embed_dim)

        self.compressor1 = Compressor(
            embed_dim=speech_dim,
            num_heads=num_heads,
            num_query=stage_tokens[0],
            n_ctx=1500,
        )
        self.stage1 = Downsampler(
            embed_dim=speech_dim
        )
        self.compressor2 = Compressor(
            embed_dim=speech_dim,
            num_heads=num_heads,
            num_query=stage_tokens[1],
            n_ctx=750,
        )
        self.stage2 = Downsampler(
            embed_dim=speech_dim
        )
        self.compressor3 = Compressor(
            embed_dim=speech_dim,
            num_heads=num_heads,
            num_query=stage_tokens[2],
            n_ctx=375,
        )

        self.compressor = Compressor(
            embed_dim=speech_dim,
            num_heads=num_heads,
            num_query=compression_size,
            n_ctx=stage_tokens[0] + stage_tokens[1] + stage_tokens[2]
        )

        self.out_attn = nn.ModuleList([
            Attention(
            embed_dim=speech_dim,
            num_heads=num_heads
        )
        for _ in range(depths)
        ])
    
    def embed_audio(self, mel: torch.Tensor):
        return self.whisper(mel)
        
    def forward(self, wav):
        if len(wav) <= 1:  
            speech_token = self.process_audio_for_llm_input(wav)
            speech_attn_mask = torch.zeros(1,speech_token.size(1)).bool().to(self.device)
            return speech_token, speech_attn_mask
        else:
            speech_features = []
            speech_attn_mask = []
            for w in wav:
                speech_feature = self.process_audio_for_llm_input(w)
                speech_features.append(speech_feature)
                speech_attn_mask.append(torch.zeros(1,speech_feature.size(1)).bool())

            speech_features = self.pad_sequence(speech_features,padding_side='right',padding_value=0.0).to(self.device)
            speech_attn_mask = self.pad_sequence(speech_attn_mask,padding_side='right',padding_value=True).to(self.device).squeeze(1)
            return speech_features, speech_attn_mask
    
    
    def pad_or_trim(self, array, length: int = 480000, *, axis: int = -1):
        """
        Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
        """
        if torch.is_tensor(array):
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
        else:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
        return array
    
    def process_audio_for_llm_input(self, wav):
        min_length = 16000
        n_frames = 3000
        wav = wav.flatten()

        if wav.shape[0] < min_length:
            wav = F.pad(wav, (0, min_length - wav.shape[0]))
            
        mels = whisper.log_mel_spectrogram(wav, n_mels=self.mel_dim).unsqueeze(0).to(self.device)


        # Split the audio into non-overlapping windows
        if mels.shape[-1] > n_frames:
            mel_segments = []
            for i in range(0, mels.shape[-1], n_frames):
                mel = mels[:,:,i:i+n_frames]
                if mel.shape[-1] < n_frames:
                    mel = self.pad_or_trim(mel,n_frames)
                mel_segments.append(mel)

            audio_features = torch.cat(mel_segments) 
            audio_features = self.embed_audio(audio_features)

            B, T, C = audio_features.shape
            stage_1_token = self.compressor1(x=audio_features)

            stage_1_feature = self.stage1(audio_features.transpose(1,2))
            stage_2_token = self.compressor2(x=stage_1_feature)

            stage_2_feature = self.stage2(stage_1_feature.transpose(1,2))
            stage_3_token = self.compressor3(x=stage_2_feature)

            stage_tokens = torch.cat([
                stage_1_token,stage_2_token,stage_3_token
            ],dim=1)

            compressed_tokens = self.compressor(stage_tokens)

            h_audio_feature = torch.cat([
                audio_features,stage_1_feature,stage_2_feature
            ],dim=1)

            for block in self.out_attn:
                compressed_tokens = block(
                    x=compressed_tokens,
                    xa=h_audio_feature
                )

            # Aggregate the results from all segments
            speech_tokens = self.llm_proj(compressed_tokens)
            speech_tokens = speech_tokens.view(1, B * self.compression_size, self.embed_dim)
            return speech_tokens
        else:
            mels = self.pad_or_trim(mels,3000)
            audio_feature = self.embed_audio(mels)
            stage_1_token = self.compressor1(x=audio_feature)

            stage_1_feature = self.stage1(audio_feature.transpose(1,2))
            stage_2_token = self.compressor2(x=stage_1_feature)

            stage_2_feature = self.stage2(stage_1_feature.transpose(1,2))
            stage_3_token = self.compressor3(x=stage_2_feature)

            stage_tokens = torch.cat([
                stage_1_token,stage_2_token,stage_3_token
            ],dim=1)

            compressed_tokens = self.compressor(stage_tokens)

            h_audio_feature = torch.cat([
                audio_feature,stage_1_feature,stage_2_feature
            ],dim=1)

            for block in self.out_attn:
                compressed_tokens = block(
                    x=compressed_tokens,
                    xa=h_audio_feature
                )
            speech_token = self.llm_proj(compressed_tokens)
            return speech_token         
    
    def pad_sequence(self, sequences, padding_side='right', padding_value=0.0):
        """
        Pad a list of 2D/3D tensors to the same length in dim=1 (time).
        sequences: list of tensors in shape [B, T, D] or [T, D]
        """
        assert padding_side in ['right', 'left']
        # Handle [T, D] shape by adding batch dim
        if sequences[0].ndim > 2:
            batch_size = len(sequences)
            max_len = max(seq.size(1) for seq in sequences)
            feat_dim = sequences[0].size(2)

            output = torch.full(
                (batch_size, max_len, feat_dim),
                padding_value, dtype=sequences[0].dtype, device=sequences[0].device
            )

            for i, seq in enumerate(sequences):
                length = seq.size(1)
                if padding_side == 'right':
                    output[i, :length, :] = seq
                else:
                    output[i, -length:, :] = seq
        else:
            batch_size = len(sequences)
            max_len = max(seq.size(1) for seq in sequences)
            output = torch.full(
                (batch_size, max_len), 
                padding_value, dtype=sequences[0].dtype, device=sequences[0].device
            )
            for i, seq in enumerate(sequences):
                length = seq.size(1)
                if padding_side == 'right':
                    output[i, :length] = seq
                else:
                    output[i, -length:] = seq

        return output


class FastSLM(nn.Module):
    def __init__(
        self,
        embed_dim,
        speech_dim,
        lora=True,
        lora_r=16,
        lora_a=64,
        stage_tokens=[80,80,80],
        compression=True,
        compression_size=50,
        llm_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        audio_token=['<|AUDIO|>','<|audio_bos|>','<|audio_eos|>'],
        model_name='Qwen/Qwen3-4B'
        ):
        super(FastSLM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_dim = embed_dim
        self.speech_dim = speech_dim
        self.lora = lora
        self.encoder = SpeechEncoder(
            embed_dim=embed_dim,
            speech_dim=speech_dim,
            stage_tokens=stage_tokens,
            compression=compression,
            compression_size=compression_size
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        task_token = ['<|ASR|>','<|AST|>','<|SSUM|>','<|SQQA|>']
        language_token = [f"<|{lang.upper()}|>" for lang in LANGUAGES]
        special_token = audio_token + language_token + task_token
        self.tokenizer.add_special_tokens({"additional_special_tokens":special_token})

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            _attn_implementation='sdpa',
            trust_remote_code=True
        )
        self.llm.generation_config.think_mode = False
        if self.lora:
            llm_lora_config = LoraConfig(
            r=lora_r,           
            lora_alpha=lora_a,
            target_modules=llm_modules,
            lora_dropout=0.01,  
            task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, llm_lora_config)

    def process_audio(self, audio_array, sample_rate):
        # Resample only if necessary
        audio = torch.tensor(audio_array, dtype=torch.float32)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(audio)
        return audio
    
    def forward(
        self, 
        audio,
        input_ids,
        labels,
    ):
        B = len(audio)
        speech_query, speech_attn_mask  = self.encoder(audio)
        # speech_attn_mask = (speech_attn_mask.int() <= 0).int()
        # text embedding
        token_embedding = self.llm.get_input_embeddings()
        speech_label_len = int(speech_query.shape[1])
        speech_labels = torch.full(
            (speech_query.shape[0], speech_label_len),
            fill_value=-100,
            dtype=torch.long,
            device=speech_query.device
        )

        audio_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        idx = torch.nonzero(input_ids[0] == audio_token_id)[0][0].item()
        left_token, right_token = input_ids[:,:idx], input_ids[:,idx+1:]

        left_label, right_label = labels[:,:idx], labels[:,idx+1:]
        left_embed = token_embedding(left_token.long()).to(speech_query.device)
        right_embed = token_embedding(right_token.long()).to(speech_query.device)

        left_mask = (left_token != self.tokenizer.pad_token_id).long().to(self.device)
        right_mask = (right_token != self.tokenizer.pad_token_id).long().to(self.device)
        speech_attn_mask = (speech_attn_mask.int() <= 0).long()

        input_embeds = torch.cat([left_embed,speech_query,right_embed],dim=1)
        labels = torch.cat([left_label,speech_labels,right_label], dim=1).long()
        attention_masks = torch.cat([
            left_mask, speech_attn_mask, right_mask
            ], dim=1
        ) 

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_masks,
            labels=labels
        )
    
        logits = outputs.logits
        loss = outputs.loss

        return loss, logits, labels
    
    def generate(
        self,
        input_ids,
        audio=None,
        wav_path=None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        top_k: int = 20,
        top_p: float = 0.95,
        temperature: float = 0.2,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        use_cache: bool = True
    ):
        token_embedding = self.llm.get_input_embeddings()
        audio_token_id = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")

        # audio feature processing
        if wav_path or audio is not None:
            B = len(audio)
            
            if wav_path and not audio:
                audio, sample_rate = torchaudio.load(wav_path)
                audio = self.process_audio(audio,sample_rate)
                
            speech_query, speech_attn_mask  = self.encoder(audio)

            idx = torch.nonzero(input_ids[0] == audio_token_id)[0][0].item()
            left_token, right_token = input_ids[:,:idx], input_ids[:,idx+1:]

            left_embed = token_embedding(left_token.long()).to(speech_query.device)
            right_embed = token_embedding(right_token.long()).to(speech_query.device)
            
            left_mask = (left_token != self.tokenizer.pad_token_id).long().to(self.device)
            right_mask = (right_token != self.tokenizer.pad_token_id).long().to(self.device)
            speech_attn_mask = (speech_attn_mask.int() <= 0).long()

            input_embeds = torch.cat([left_embed,speech_query,right_embed],dim=1)
            attention_masks = torch.cat([
                left_mask,speech_attn_mask,right_mask], dim=1
            ) 
            generated_ids = self.llm.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_masks,  
                use_cache=use_cache,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,  
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            input_embeds = token_embedding(input_ids)
            attention_masks = torch.ones([
                input_embeds.size(0), input_embeds.size(1)], dtype=torch.long, device=input_embeds.device
            )
            with self.llm.disable_adapter():
                generated_ids = self.llm.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_masks,  
                use_cache=use_cache,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,  
                eos_token_id=self.tokenizer.eos_token_id
            )
        # generated_ids = generated_ids[:,input_embeds.shape[1]:].tolist() 
        output_text = self.tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=True, add_special_tokens=False, skip_special_tokens=True)
        return output_text
        
    def pad_sequence(self, sequences, padding_side='right', padding_value=0.0):
        """
        Pad a list of 2D/3D tensors to the same length in dim=1 (time).
        sequences: list of tensors in shape [B, T, D] or [T, D]
        """
        assert padding_side in ['right', 'left']
        # Handle [T, D] shape by adding batch dim
        if sequences[0].ndim > 2:
            batch_size = len(sequences)
            max_len = max(seq.size(1) for seq in sequences)
            feat_dim = sequences[0].size(2)

            output = torch.full(
                (batch_size, max_len, feat_dim),
                padding_value, dtype=sequences[0].dtype, device=sequences[0].device
            )

            for i, seq in enumerate(sequences):
                length = seq.size(1)
                if padding_side == 'right':
                    output[i, :length, :] = seq
                else:
                    output[i, -length:, :] = seq
        else:
            batch_size = len(sequences)
            max_len = max(seq.size(1) for seq in sequences)
            output = torch.full(
                (batch_size, max_len), 
                padding_value, dtype=sequences[0].dtype, device=sequences[0].device
            )
            for i, seq in enumerate(sequences):
                length = seq.size(1)
                if padding_side == 'right':
                    output[i, :length] = seq
                else:
                    output[i, -length:] = seq

        return output








        




