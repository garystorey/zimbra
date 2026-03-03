# Talking Head Video Generator — Technical Design Document

## 1. System Overview

```mermaid
graph TB
    subgraph "User Interface"
        CLI[generate.py / generate.bat]
    end

    subgraph "Orchestration"
        Pipeline[pipeline.py]
    end

    subgraph "Engines"
        TTS[tts_engine.py<br/>Chatterbox Turbo]
        Video[video_engine.py<br/>InfiniteTalk]
    end

    subgraph "Infrastructure"
        Config[config.py]
        ModelMgr[model_manager.py]
        GGUF[gguf_loader.py]
        Utils[utils.py]
    end

    subgraph "Vendor Code"
        IT[vendor/infinitetalk/<br/>garystorey/InfiniteTalk fork]
    end

    subgraph "External"
        HF[HuggingFace Hub]
        FFmpeg[FFmpeg]
    end

    CLI --> Pipeline
    Pipeline --> TTS
    Pipeline --> Video
    Video --> GGUF
    Video --> IT
    TTS --> ModelMgr
    Video --> ModelMgr
    ModelMgr --> HF
    Video --> FFmpeg
    Utils --> FFmpeg
    Config --> Pipeline
    Config --> TTS
    Config --> Video
```

---

## 2. Pipeline Architecture

### 2.1 Full Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as generate.py
    participant Pipe as pipeline.py
    participant TTS as TTSEngine
    participant VE as VideoEngine
    participant MM as model_manager
    participant GPU as GPU (16 GB)

    User->>CLI: generate.bat
    CLI->>CLI: setup_logging(), _check_gpu()
    CLI->>Pipe: Pipeline.run()

    Note over Pipe: Pre-flight downloads
    Pipe->>MM: preflight_downloads()
    MM-->>Pipe: all files present

    Note over Pipe: Phase 1: TTS
    Pipe->>TTS: load_model()
    TTS->>GPU: Load Chatterbox (~4 GB)
    Pipe->>TTS: generate(text, voice_ref)
    TTS-->>Pipe: audio.wav
    Pipe->>TTS: unload_model()
    TTS->>GPU: Free to 0 GB

    Note over Pipe: Phase 2: Video
    Pipe->>VE: load_model()
    VE->>MM: ensure_infinitetalk_models()
    VE->>MM: ensure_gguf_models()
    VE->>MM: ensure_lora_models()
    VE->>GPU: Load GGUF model (~12.8 GB)
    VE->>VE: pre_encode_text() on GPU
    VE->>VE: block_swap(22 persistent, 18 swap)
    VE->>VE: apply_lora_as_patches()
    Pipe->>VE: generate(image, audio)
    VE-->>Pipe: video.mp4
    Pipe->>VE: unload_model()
    VE->>GPU: Free to 0 GB

    Pipe-->>CLI: {audio_path, video_path, timings}
    CLI-->>User: Done! output/*.mp4
```

### 2.2 VRAM Timeline

```mermaid
gantt
    title GPU VRAM Usage Over Time
    dateFormat X
    axisFormat %s

    section TTS
    Load Chatterbox (~4 GB)     :a1, 0, 15
    Generate audio              :a2, 15, 25
    Unload (0 GB)               :milestone, m1, 25, 0

    section Video Load
    Load GGUF DiT (~12.8 GB)    :b1, 26, 80
    Pre-encode T5 on GPU        :b2, 80, 90
    Offload VAE/CLIP to CPU     :b3, 90, 95
    Block swap setup            :b4, 95, 100

    section Video Generate
    Chunk 1 denoise (~8.5 GB)   :c1, 100, 240
    Chunk 1 VAE decode          :c2, 240, 268
    Chunk 2 denoise             :c3, 268, 408
    Chunk 2 VAE decode          :c4, 408, 436
    Chunk N...                  :c5, 436, 600

    section Cleanup
    Unload (0 GB)               :milestone, m2, 600, 0
```

---

## 3. Component Design

### 3.1 TTS Engine (`tts_engine.py`)

```mermaid
flowchart TD
    A[Input: text + voice_ref] --> B[normalize_text]
    B --> B2["Strip quotes, smart quotes → ASCII"]
    B2 --> C[_split_script_to_segments by lines]
    C --> D{Segment type?}
    D -->|text| E[split_into_sentences at .!? boundaries]
    D -->|pause| F[Generate silence array]
    D -->|audio| F2["Load & resample audio file to 24kHz"]
    E --> G[generate_chunk per sub-sentence]
    G --> G2[trim_silence on each chunk]
    G2 --> H[Join sub-sentences with 100ms gap]
    H --> I[Join lines with 200ms gap]
    F --> I
    F2 --> I
    I --> J[Pad: 0.25s front + 0.5s end]
    J --> K[Save WAV at 24 kHz]
```

**Key Implementation Details:**

- **Model**: `ChatterboxTurboTTS.from_pretrained(device="cuda")`
- **Voice ref**: Must be >5 seconds. Converted to WAV if MP3.
- **Text normalization**: Smart quotes → ASCII `"` → stripped. Em/en dashes → `-`. Prevents phantom pauses and truncation in Chatterbox.
- **`--text` CLI option**: Pass script text directly instead of reading from `inputs/scripts/default.txt`. Priority: `--text` > file.
- **`[s:N]` pause tags**: Insert N seconds of silence (supports `[s:1.5]`, `[s:.5]`, etc.)
- **`${filename}` audio references**: Splice pre-recorded audio files from `inputs/scripts/` into the output. Files are loaded and resampled to 24kHz.
- **Two-level splitting**: Lines → sub-sentences. Each sub-sentence is a separate TTS call. 100ms gap within lines, 200ms between lines.
- **Silence trimming**: Leading (50ms margin) and trailing (200ms margin) trimmed at -40dB threshold per chunk.
- **Sampling**: `temperature=0.7`, `top_k=150`, `top_p=0.92`, `repetition_penalty=1.2`. Temperature <0.5 causes looping.
- **Output**: 24 kHz mono WAV

### 3.2 Video Engine (`video_engine.py`)

```mermaid
flowchart TD
    A[load_model] --> B[Download models if needed]
    B --> C[Load wav2vec2 audio encoder to CPU]
    C --> D[Create InfiniteTalkPipeline]
    D --> E[Load GGUF weights via gguf_loader]
    E --> F[Pre-encode T5 text prompt on GPU]
    F --> G[Offload VAE/CLIP/T5 to CPU]
    G --> H[Auto-tune block swap]
    H --> I[Apply LightX2V LoRA as patches]
    I --> J[Model ready]

    K[generate] --> L[Normalize audio to -23 LUFS]
    L --> M[Extract wav2vec2 embeddings]
    M --> N[Build input_clip dict]
    N --> O[Call pipe.generate_infinitetalk]
    O --> P[FFmpeg mux video + audio]
    P --> Q[Output MP4]
```

### 3.3 GGUF Loader (`gguf_loader.py`)

```mermaid
flowchart TD
    A[load_gguf_weights] --> B[load_gguf_state_dict: base GGUF]
    A --> C[load_gguf_state_dict: adapter GGUF]
    B --> D[Merge state dicts]
    C --> D
    D --> E[load_gguf_into_model]
    E --> F{Weight type?}
    F -->|Linear| G[Store packed bytes as nn.Parameter<br/>Set _gguf_qtype, _gguf_shape on module]
    F -->|Other| H[Dequantize at load time<br/>Store as float/bf16]
    G --> I[patch_gguf_linears]
    I --> J[Monkey-patch Linear.forward:<br/>1. Dequantize packed weight → bf16<br/>2. Apply LoRA if present<br/>3. F.linear]

    K[apply_lora_as_patches] --> L[Load LoRA safetensors]
    L --> M[Match LoRA keys to model modules]
    M --> N[Register _lora_up, _lora_down, _lora_scale<br/>as buffers on each Linear module]
```

### 3.4 Block Swap Architecture

```mermaid
flowchart LR
    subgraph "GPU (16 GB)"
        Persistent["Blocks 0-21<br/>(persistent, ~7 GB)"]
        SwapBuf["Swap Buffer<br/>(1 block, ~320 MB)"]
        Activations["Activation Headroom<br/>(~8.2 GB)"]
    end

    subgraph "CPU (Pinned Memory)"
        Swapped["Blocks 22-39<br/>(18 blocks, pinned)"]
    end

    Swapped -->|"Async prefetch<br/>(CUDA stream)"| SwapBuf
    SwapBuf -->|"Compute on<br/>default stream"| Persistent
    SwapBuf -->|"Offload after<br/>compute"| Swapped
```

**Block Swap Execution Loop:**

```mermaid
sequenceDiagram
    participant CPU as CPU (Pinned)
    participant Stream as Prefetch Stream
    participant GPU as GPU (Default Stream)

    Note over CPU,GPU: Processing swap blocks 22-39

    CPU->>Stream: Prefetch block N+1 to GPU (async)
    GPU->>GPU: Compute block N (default stream)
    GPU->>GPU: sync prefetch stream
    GPU->>CPU: Offload block N to CPU (sync)
    CPU->>CPU: Re-pin block N memory

    Note over CPU,GPU: Repeat for each swap block
```

---

## 4. Denoising Loop (Vendor Patch)

### 4.1 DPM++ SDE Solver

```mermaid
flowchart TD
    A[Start: noisy latent x_T] --> B[For each timestep t]
    B --> C{skip_cfg?<br/>text_cfg≈1 AND audio_cfg≈1}
    C -->|Yes| D[1 forward pass: noise_pred = model x_t, audio]
    C -->|No| E[Pass 1: noise_pred_cond = model x_t, audio]
    E --> F[Pass 2: noise_pred_uncond = model x_t, drop_audio]
    F --> G[CFG: noise_pred = uncond + audio_cfg × cond - uncond]
    D --> H[DPM++ SDE update]
    G --> H
    H --> I["x_t-1 = drift(x_t, denoised) + diffusion(noise)"]
    I --> J{More steps?}
    J -->|Yes| B
    J -->|No| K[Inject motion frames from previous chunk]
    K --> L[Save motion latent BEFORE VAE decode]
    L --> M[VAE decode → pixel frames]
```

**SDE Update Formula:**

```
alpha_s, alpha_t = 1 - sigma_s, 1 - sigma_t
h = log(alpha_t/sigma_t) - log(alpha_s/sigma_s)
x_t = (sigma_t/sigma_s × exp(-h)) × x_s
    + (alpha_t × (1 - exp(-2h))) × denoised
    + sigma_t × sqrt(1 - exp(-2h)) × noise
```

### 4.2 Streaming Chunk Pipeline

```mermaid
flowchart TD
    A[Audio: 8.2s → 206 frames] --> B[Chunk 1: frames 0-80]
    B --> C[Denoise 6 steps]
    C --> D[Save motion latent frames 68-80]
    D --> E[Offload DiT blocks]
    E --> F[VAE decode → pixels]
    F --> G[Offload VAE, restore DiT blocks]
    G --> H[Chunk 2: frames 68-148]
    H --> I[Inject motion latent as frames 0-12]
    I --> J[Denoise 6 steps]
    J --> K[Save motion latent frames 68-80]
    K --> L[Continue until all audio covered]

    style D fill:#ff9,stroke:#333
    style I fill:#ff9,stroke:#333
```

**Key parameters:**

- `frame_num=81` — frames per chunk (4n+1)
- `motion_frame=13` — overlap frames between chunks (0.52s)
- New frames per chunk: 81 - 13 = 68 (2.72s at 25 FPS)
- Direct latent passthrough (no VAE round-trip between chunks)

---

## 5. Installation Architecture

### 5.1 Dependency Resolution

```mermaid
flowchart TD
    A[install.bat] --> B[install.py]
    B --> C[Find Python 3.10-3.12]
    C --> D[Create venv/]
    D --> E[Upgrade pip, setuptools, wheel]
    E --> F["torch==2.4.1+cu121<br/>torchvision, torchaudio<br/>(from cu121 index)"]
    F --> G["xformers==0.0.28.post1<br/>(from cu124 index)"]
    G --> H["flash_attn==2.7.4.post1<br/>(prebuilt Windows wheel)"]
    H --> I["numpy==1.26.4<br/>(pin BEFORE requirements.txt)"]
    I --> J[requirements.txt<br/>InfiniteTalk deps]
    J --> K["chatterbox-tts --no-deps"]
    K --> L[Chatterbox unique deps<br/>librosa, spacy-pkuseg, etc.]
    L --> M[git clone garystorey/InfiniteTalk → vendor/]
    M --> N[Create runtime directories]
    N --> O[Verify all imports]

    style F fill:#f96,stroke:#333
    style G fill:#f96,stroke:#333
    style H fill:#f96,stroke:#333
    style K fill:#9cf,stroke:#333
```

**Critical ordering constraints:**

1. PyTorch must be installed first (cu121 index)
2. xformers from cu124 index (cu121 lacks Windows wheels for torch 2.4.1)
3. numpy pinned before requirements.txt (prevents PyTorch pulling numpy 2.x)
4. Chatterbox with `--no-deps` to avoid torch/diffusers conflicts

### 5.2 Model Download Flow

```mermaid
flowchart TD
    A[First generate.bat run] --> B[Pre-flight download check]
    B --> C{All files present?}
    C -->|Yes| D[Skip downloads]
    C -->|No| E[Show download progress bar]

    E --> F[Wan2.1 VAE+CLIP ~5.3 GB<br/>snapshot_download]
    E --> G[wav2vec2 ~360 MB<br/>snapshot_download]
    E --> H[GGUF base Q4_0 ~10.2 GB<br/>hf_hub_download]
    E --> I[GGUF adapter Q8 ~2.65 GB<br/>hf_hub_download]
    E --> J[LoRA I2V rank-64 ~738 MB<br/>hf_hub_download]

    F --> K[All downloads complete]
    G --> K
    H --> K
    I --> K
    J --> K
    K --> D
    D --> L[Continue to pipeline]
```

---

## 6. VRAM Management Strategy

### 6.1 Memory Budget (16 GB GPU)

```
┌─────────────────────────────────────────────┐
│                 16 GB VRAM                   │
├─────────────────────────────────────────────┤
│ Persistent DiT blocks (22 × 320 MB) = 7.0 GB│
│ Swap buffer (1 × 320 MB)           = 0.3 GB│
│ Non-block params                    = 0.5 GB│
│ ─────────────────────────────────────────── │
│ Activation headroom                 = 8.2 GB│
│   ├─ GGUF dequant temporaries (bf16)        │
│   ├─ QKV projections                        │
│   ├─ Attention maps                         │
│   └─ FFN intermediates                      │
└─────────────────────────────────────────────┘
```

### 6.2 Critical VRAM Events

```mermaid
stateDiagram-v2
    [*] --> TTS_Loaded: Load Chatterbox (~4 GB)
    TTS_Loaded --> GPU_Empty: Unload TTS (0 GB)
    GPU_Empty --> GGUF_Loaded: Load GGUF model
    GGUF_Loaded --> T5_Encoded: Pre-encode T5 on GPU
    T5_Encoded --> Offloaded: Offload VAE/CLIP/T5 to CPU
    Offloaded --> BlockSwap: block_swap(18)
    BlockSwap --> Denoising: 22 persistent + 18 swap

    state Denoising {
        [*] --> ForwardPass: ~8.5 GB allocated
        ForwardPass --> EmptyCache: empty_cache() after each pass
        EmptyCache --> ForwardPass: Next pass
        ForwardPass --> OffloadBlocks: All steps done
    }

    Denoising --> VAE_Decode: Offload DiT, load VAE
    VAE_Decode --> RestoreBlocks: VAE.cpu(), restore DiT
    RestoreBlocks --> Denoising: Next chunk

    VAE_Decode --> [*]: Last chunk → unload all
```

---

## 7. Logging Architecture

```mermaid
flowchart TD
    A[logging.debug/info/warning] --> B{Root Logger<br/>Level: DEBUG}
    B --> C[File Handler<br/>Level: DEBUG<br/>output/*.log]
    B --> D[No console logging handler]

    G[Vendor print/tqdm] --> J[Redirected to devnull]

    K[Rich Progress Bars] --> L[Real stderr<br/>config._REAL_STDERR]
```

**Key design decision:** Root logger is ALWAYS DEBUG. All debug/info messages go to the log file. Console shows only Rich progress bars — no logging handler is attached to the console. This ensures full diagnostic detail in the log file while keeping the UI clean.

---

## 8. Error Handling

```mermaid
flowchart TD
    A[Pipeline.run] --> B{TTS Phase}
    B -->|OOM| C[Catch OutOfMemoryError]
    C --> D[_cleanup → unload all models]
    D --> E[Raise RuntimeError with suggestion]

    B -->|Success| F{Video Phase}
    F -->|OOM| C
    F -->|Success| G[Save outputs]

    H[KeyboardInterrupt] --> D

    I[Any Exception] --> J[_cleanup guaranteed via try/finally]
    J --> K[Log exception to file]
    K --> L[Print user-friendly error to console]
```

---

## 9. File Dependencies

```mermaid
graph TD
    generate.py --> config.py
    generate.py --> pipeline.py
    pipeline.py --> config.py
    pipeline.py --> utils.py
    pipeline.py --> model_manager.py
    pipeline.py --> tts_engine.py
    pipeline.py --> video_engine.py
    tts_engine.py --> config.py
    tts_engine.py --> utils.py
    video_engine.py --> config.py
    video_engine.py --> model_manager.py
    video_engine.py --> gguf_loader.py
    model_manager.py --> config.py
    gguf_loader.py --> config.py
    video_engine.py --> vendor_infinitetalk["vendor/infinitetalk/"]

    style vendor_infinitetalk fill:#fdd,stroke:#c33
```

---

## 10. Vendor Patches Summary

Patches maintained in the [garystorey/InfiniteTalk](https://github.com/garystorey/InfiniteTalk) fork. `install.py` clones this fork into `vendor/infinitetalk/` — no post-clone patching needed.

Patches that are **essential** for correct operation:

| File                             | Patch                                             | Purpose                                    |
| -------------------------------- | ------------------------------------------------- | ------------------------------------------ |
| `wan/modules/multitalk_model.py` | `block_swap()`, `_run_blocks_with_swap()`         | Block-level swap with async prefetch       |
| `wan/modules/multitalk_model.py` | `human_num` param in `WanSelfAttention.forward()` | Skip O(n²) attention map for single-person |
| `wan/multitalk.py`               | DPM++ SDE solver implementation                   | Stochastic solver for streaming chunks     |
| `wan/multitalk.py`               | `_skip_cfg` optimization                          | 1 pass/step when both CFG scales ≈ 1.0     |
| `wan/multitalk.py`               | Direct latent passthrough                         | Save motion latent BEFORE VAE decode       |
| `wan/multitalk.py`               | `torch.cuda.empty_cache()` between passes         | Prevent GGUF dequant VRAM fragmentation    |
| `wan/multitalk.py`               | Block offload before VAE decode                   | Free VRAM for 3D transposed convolutions   |
| `wan/multitalk.py`               | VAE offload after decode                          | Free VAE VRAM for next chunk denoising     |
| `wan/multitalk.py`               | Bilinear resize (was nearest)                     | Better interpolation quality               |
| `wan/utils/multitalk_utils.py`   | FFmpeg CRF 10→1                                   | Near-lossless intermediate encoding        |
| `wan/utils/multitalk_utils.py`   | Remove torch_gc() + @torch.compile                | Eliminate hot-path overhead                |
