import gc
import os
import platform
import psutil
import tempfile
from glob import glob
import traceback
import sys
import click
import gradio as gr
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from cached_path import cached_path

# Ensure the project root (the directory that contains the `lemas_tts` package)
# is on sys.path so that `import lemas_tts...` works regardless of CWD.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # .../LEMAS-TTS/lemas_tts/scripts -> REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lemas_tts.api import TTS, PRETRAINED_ROOT, CKPTS_ROOT

# Device detection for TTS main model. For inference_gradio we always run
# so this global `device` is mostly informational.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# HF location for large TTS checkpoints
HF_PRETRAINED_ROOT = "hf://LEMAS-Project/LEMAS-TTS/pretrained_models"

# 指向 `pretrained_models` 里的 espeak-ng-data（本地自带的字典）
# 动态库交给系统安装的 espeak-ng 来提供（通过 apt），不强行指定 PHONEMIZER_ESPEAK_LIBRARY，
# 避免本地复制的 .so 与基础镜像不兼容。
ESPEAK_DATA_DIR = Path(PRETRAINED_ROOT) / "espeak-ng-data"
os.environ["ESPEAK_DATA_PATH"] = str(ESPEAK_DATA_DIR)
os.environ["ESPEAKNG_DATA_PATH"] = str(ESPEAK_DATA_DIR)


class UVR5:
    """Small wrapper around the bundled uvr5 implementation for denoising."""

    def __init__(self, model_dir):
        # Code directory is always the top-level `uvr5` folder in this repo
        # (i.e., /path/to/LEMAS-TTS/uvr5), which contains multiprocess_cuda_infer.py
        self.code_dir = os.path.join(str(REPO_ROOT), "uvr5")
        self.model_dir = model_dir
        self.model = self.load_model(device)
    
    def load_model(self, device="cpu"):
        import sys, json, os, torch
        if self.code_dir not in sys.path:
            sys.path.append(self.code_dir)

        from multiprocess_cuda_infer import ModelData, Inference
        # In the minimal LEMAS-TTS layout, UVR5 weights live under:
        model_path = os.path.join(self.model_dir, "Kim_Vocal_1.onnx")
        config_path = os.path.join(self.model_dir, "MDX-Net-Kim-Vocal1.json")
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        model_data = ModelData(
            model_path=model_path,
            audio_path=self.model_dir,
            result_path=self.model_dir,
            device=device,
            process_method="MDX-Net",
            base_dir=self.model_dir,
            **configs,
        )

        uvr5_model = Inference(model_data, 'cpu')
        uvr5_model.load_model(model_path, 1)
        return uvr5_model
        
    def denoise(self, audio_info):
        print("denoise UVR5: ", audio_info)
        input_audio = load_wav(audio_info, sr=44100, channel=2)
        output_audio = self.model.demix_base({0:input_audio.squeeze()}, is_match_mix=False)
        # transform = torchaudio.transforms.Resample(44100, 16000)
        # output_audio = transform(output_audio)
        return output_audio.squeeze().T.numpy(), 44100


denoise_model = UVR5(
    model_dir=Path(PRETRAINED_ROOT) / "uvr5",
)

def load_wav(audio_info, sr=16000, channel=1):
    print("load audio:", audio_info)
    audio, raw_sr = torchaudio.load(audio_info)
    audio = audio.T if len(audio.shape) > 1 and audio.shape[1] == 2 else audio
    audio = audio / torch.max(torch.abs(audio))
    audio = audio.squeeze().float()
    if channel == 1 and len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(dim=0, keepdim=True)
    elif channel == 2 and len(audio.shape) == 1:
        audio = torch.stack((audio, audio)) # mono to stereo
    if raw_sr != sr:
        audio = torchaudio.functional.resample(audio.squeeze(), raw_sr, sr)
    audio = torch.clip(audio, -0.999, 0.999).squeeze()
    return audio


def denoise(audio_info):
    # Return a numpy waveform tuple for direct playback in Gradio.
    denoised_audio, sr = denoise_model.denoise(audio_info)
    return (sr, denoised_audio)

def cancel_denoise(audio_info):
    return audio_info


def get_checkpoints_project(project_name=None, is_gradio=True):
    """Get available checkpoint files"""
    checkpoint_dir = [str(CKPTS_ROOT)]
    # Remote ckpt locations on HF (used when local ckpts are not present)
    remote_ckpts = {
        "multilingual_grl": f"{HF_PRETRAINED_ROOT}/ckpts/multilingual_grl/multilingual_grl.safetensors",
        "multilingual_prosody": f"{HF_PRETRAINED_ROOT}/ckpts/multilingual_prosody/multilingual_prosody.safetensors",
    }

    if project_name is None:
        # Look for checkpoints in local directory
        files_checkpoints = []
        for path in checkpoint_dir:
            if os.path.isdir(path):
                files_checkpoints.extend(glob(os.path.join(path, "**/*.pt"), recursive=True))
                files_checkpoints.extend(glob(os.path.join(path, "**/*.safetensors"), recursive=True))
                break
        # Fallback to remote ckpts if none found locally
        if not files_checkpoints:
            files_checkpoints = list(remote_ckpts.values())
    else:
        files_checkpoints = []
        if os.path.isdir(checkpoint_dir[0]):
            files_checkpoints = glob(os.path.join(checkpoint_dir[0], project_name, "*.pt"))
            files_checkpoints.extend(glob(os.path.join(checkpoint_dir[0], project_name, "*.safetensors")))
        # If no local ckpts for this project, try remote mapping
        if not files_checkpoints:
            ckpt = remote_ckpts.get(project_name)
            files_checkpoints = [ckpt] if ckpt is not None else []
    print("files_checkpoints:", project_name, files_checkpoints)
    # Separate pretrained and regular checkpoints
    pretrained_checkpoints = [f for f in files_checkpoints if "pretrained_" in os.path.basename(f)]
    regular_checkpoints = [
        f
        for f in files_checkpoints
        if "pretrained_" not in os.path.basename(f) and "model_last.pt" not in os.path.basename(f)
    ]

    # Sort regular checkpoints by number
    try:
        regular_checkpoints = sorted(
            regular_checkpoints, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
    except (IndexError, ValueError):
        regular_checkpoints = sorted(regular_checkpoints)

    # Combine in order: pretrained, regular, last
    files_checkpoints = pretrained_checkpoints + regular_checkpoints

    select_checkpoint = None if not files_checkpoints else files_checkpoints[-1]

    if is_gradio:
        return gr.update(choices=files_checkpoints, value=select_checkpoint)

    return files_checkpoints, select_checkpoint


def get_available_projects():
    """Get available project names from data directory"""
    data_paths = [
        str(Path(PRETRAINED_ROOT) / "data"),
    ]
    
    project_list = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            for folder in os.listdir(data_path):
                path_folder = os.path.join(data_path, folder)
                if "test" not in folder:
                    project_list.append(folder)
            break
    # Fallback: if no local data dir, default to known HF projects
    if not project_list:
        project_list = ["multilingual_grl", "multilingual_prosody"]
    project_list.sort(reverse=True)
    print("project_list:", project_list)
    return project_list

def infer(
    project, file_checkpoint, exp_name, ref_text, ref_audio, denoise_audio, gen_text, nfe_step, use_ema, separate_langs, frontend, speed, cfg_strength, use_acc_grl, ref_ratio, no_ref_audio, sway_sampling_coef, use_prosody_encoder, seed
):
    global tts_api

    # Resolve checkpoint path (local or HF URL)
    ckpt_path = file_checkpoint
    if isinstance(ckpt_path, str) and ckpt_path.startswith("hf://"):
        try:
            ckpt_resolved = str(cached_path(ckpt_path))
        except Exception as e:
            traceback.print_exc()
            return None, f"Error downloading checkpoint: {str(e)}", ""
    else:
        ckpt_resolved = ckpt_path

    if not os.path.isfile(ckpt_resolved):
        return None, "Checkpoint not found!", ""

    # Prepare reference audio:
    # - `ref_audio` from Gradio is a filepath (original reference)
    # - `denoise_audio` is an optional (sr, numpy_array) tuple from UVR5.
    #   If provided, dump it to a temporary wav file and use that as ref_file.
    ref_audio_path = ref_audio
    tmp_ref_path = None
    if denoise_audio is not None:
        try:
            sr_d, wav_d = denoise_audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f_ref:
                sf.write(f_ref.name, wav_d, int(sr_d), format="wav", subtype="PCM_24")
                tmp_ref_path = f_ref.name
                ref_audio_path = f_ref.name
        except Exception as e:
            traceback.print_exc()
            return None, f"Error preparing denoised reference audio: {str(e)}", ""

    # Automatically enable prosody encoder when using the prosody checkpoint
    use_prosody_encoder = True if "prosody" in str(ckpt_resolved) else False

    # Resolve vocab file (local)
    local_vocab = Path(PRETRAINED_ROOT) / "data" / project / "vocab.txt"
    if not local_vocab.is_file():
        return None, "Vocab file not found!", ""
    vocab_file = str(local_vocab)

    # Resolve prosody encoder config & weights (local)
    local_prosody_cfg = Path(CKPTS_ROOT) / "prosody_encoder" / "pretssel_cfg.json"
    local_prosody_ckpt = Path(CKPTS_ROOT) / "prosody_encoder" / "prosody_encoder_UnitY2.pt"
    if not local_prosody_cfg.is_file() or not local_prosody_ckpt.is_file():
        return None, "Prosody encoder files not found!", ""
    prosody_cfg_path = str(local_prosody_cfg)
    prosody_ckpt_path = str(local_prosody_ckpt)

    # Prefer CUDA if available, but be ready to fall back to CPU when kernels
    # are not supported on the current GPU (e.g. "no kernel image is available").
    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_tts(device_for_tts: str):
        return TTS(
            model=exp_name,
            ckpt_file=ckpt_resolved,
            vocab_file=vocab_file,
            device=device_for_tts,
            use_ema=use_ema,
            frontend=frontend,
            use_prosody_encoder=use_prosody_encoder,
            prosody_cfg_path=prosody_cfg_path,
            prosody_ckpt_path=prosody_ckpt_path,
        )

    try:
        tts_api = _build_tts(preferred_device)
    except Exception as e:
        traceback.print_exc()
        if preferred_device.startswith("cuda"):
            # If CUDA init itself fails, fall back to CPU.
            try:
                tts_api = _build_tts("cpu")
            except Exception as e2:
                traceback.print_exc()
                if tmp_ref_path is not None and os.path.isfile(tmp_ref_path):
                    os.remove(tmp_ref_path)
                return None, f"Error loading model on CUDA and CPU: {e2}", ""
        else:
            if tmp_ref_path is not None and os.path.isfile(tmp_ref_path):
                os.remove(tmp_ref_path)
            return None, f"Error loading model: {str(e)}", ""

    print("Model loaded >>", file_checkpoint, use_ema, "on", tts_api.device)

    if seed == -1:  # -1 used for random
        seed = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            try:
                # First attempt: use the current TTS device (preferably CUDA).
                tts_api.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text.strip(),
                    gen_text=gen_text.strip(),
                    nfe_step=nfe_step,
                    separate_langs=separate_langs,
                    speed=speed,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    use_acc_grl=use_acc_grl,
                    ref_ratio=ref_ratio,
                    no_ref_audio=no_ref_audio,
                    use_prosody_encoder=use_prosody_encoder,
                    file_wave=f.name,
                    seed=seed,
                )
                return f.name, f"Device: {tts_api.device}", str(tts_api.seed)
            except Exception as e:
                traceback.print_exc()
                msg = str(e)
                # If CUDA kernels are not compatible with this GPU, fall back to CPU.
                if preferred_device.startswith("cuda") and (
                    "no kernel image is available for execution on the device" in msg
                    or "CUDA error" in msg
                ):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        # Rebuild model on CPU and retry once.
                        tts_api_cpu = _build_tts("cpu")
                        tts_api_cpu.infer(
                            ref_file=ref_audio_path,
                            ref_text=ref_text.strip(),
                            gen_text=gen_text.strip(),
                            nfe_step=nfe_step,
                            separate_langs=separate_langs,
                            speed=speed,
                            cfg_strength=cfg_strength,
                            sway_sampling_coef=sway_sampling_coef,
                            use_acc_grl=use_acc_grl,
                            ref_ratio=ref_ratio,
                            no_ref_audio=no_ref_audio,
                            use_prosody_encoder=use_prosody_encoder,
                            file_wave=f.name,
                            seed=seed,
                        )
                        return f.name, f"Device: {tts_api_cpu.device}", str(tts_api_cpu.seed)
                    except Exception as e2:
                        traceback.print_exc()
                        return None, f"Inference error after CUDA->CPU fallback: {e2}", ""
                # Non-CUDA errors or CPU path failures: surface as-is.
                return None, f"Inference error: {msg}", ""
    finally:
        # Remove temporary reference file if created
        if tmp_ref_path is not None and os.path.isfile(tmp_ref_path):
            os.remove(tmp_ref_path)


def get_gpu_stats():
    """Get GPU statistics"""
    gpu_stats = ""

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)  # in GB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**2)  # in MB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**2)  # in MB

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )
    elif torch.xpu.is_available():
        gpu_count = torch.xpu.device_count()
        for i in range(gpu_count):
            gpu_name = torch.xpu.get_device_name(i)
            gpu_properties = torch.xpu.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)  # in GB
            allocated_memory = torch.xpu.memory_allocated(i) / (1024**2)  # in MB
            reserved_memory = torch.xpu.memory_reserved(i) / (1024**2)  # in MB

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )
    elif torch.backends.mps.is_available():
        gpu_count = 1
        gpu_stats += "MPS GPU\n"
        total_memory = psutil.virtual_memory().total / (
            1024**3
        )  # Total system memory (MPS doesn't have its own memory)
        allocated_memory = 0
        reserved_memory = 0

        gpu_stats += (
            f"Total system memory: {total_memory:.2f} GB\n"
            f"Allocated GPU memory (MPS): {allocated_memory:.2f} MB\n"
            f"Reserved GPU memory (MPS): {reserved_memory:.2f} MB\n"
        )

    else:
        gpu_stats = "No GPU available"

    return gpu_stats


def get_cpu_stats():
    """Get CPU statistics"""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_used = memory_info.used / (1024**2)
    memory_total = memory_info.total / (1024**2)
    memory_percent = memory_info.percent

    pid = os.getpid()
    process = psutil.Process(pid)
    nice_value = process.nice()

    cpu_stats = (
        f"CPU Usage: {cpu_usage:.2f}%\n"
        f"System Memory: {memory_used:.2f} MB used / {memory_total:.2f} MB total ({memory_percent}% used)\n"
        f"Process Priority (Nice value): {nice_value}"
    )

    return cpu_stats


def get_combined_stats():
    """Get combined system stats"""
    gpu_stats = get_gpu_stats()
    cpu_stats = get_cpu_stats()
    combined_stats = f"### GPU Stats\n{gpu_stats}\n\n### CPU Stats\n{cpu_stats}"
    return combined_stats


# Create Gradio interface
with gr.Blocks(title="LEMAS-TTS Inference") as app:
    gr.Markdown(
        """
        # Zero-Shot TTS

        Set seed to -1 for random generation.
        """
    )
    with gr.Accordion("Model configuration", open=False):
    # Model configuration
        with gr.Row():
            exp_name = gr.Radio(
                label="Model",
                choices=["multilingual_grl", "multilingual_prosody"],
                value="multilingual_grl",
                visible=False,
            )
        # Project selection
        available_projects = get_available_projects()

        # Get initial checkpoints
        list_checkpoints, checkpoint_select = get_checkpoints_project(available_projects[0] if available_projects else None, False)

        with gr.Row():
            with gr.Column(scale=1):
                # load_models_btn = gr.Button(value="Load models")
                cm_project = gr.Dropdown(
                    choices=available_projects, 
                    value=available_projects[0] if available_projects else None,
                    label="Project", 
                    allow_custom_value=True, 
                    scale=4
                )
                
            with gr.Column(scale=5):
                cm_checkpoint = gr.Dropdown(
                    choices=list_checkpoints, value=checkpoint_select, label="Checkpoints", allow_custom_value=True # scale=4, 
)
            bt_checkpoint_refresh = gr.Button("Refresh", scale=1)

        with gr.Row():
            ch_use_ema = gr.Checkbox(label="Use EMA", visible=False, value=True, scale=2, info="Turn off at early stage might offer better results")
            frontend = gr.Radio(label="Frontend", visible=False, choices=["phone", "char", "bpe"], value="phone", scale=3)
            separate_langs = gr.Checkbox(label="Separate Languages", visible=False, value=True, scale=2, info="separate language tokens")

        # Inference parameters
        with gr.Row():
            nfe_step = gr.Number(label="NFE Step", scale=1, value=64)
            speed = gr.Slider(label="Speed", scale=3, value=1.0, minimum=0.5, maximum=1.5, step=0.1)
            cfg_strength = gr.Slider(label="CFG Strength", scale=2, value=5.0, minimum=0.0, maximum=10.0, step=1)
            sway_sampling_coef = gr.Slider(label="Sway Sampling Coef", scale=2, value=3, minimum=2, maximum=5, step=0.1)
            ref_ratio = gr.Slider(label="Ref Ratio", scale=2, value=1.0, minimum=0.0, maximum=1.0, step=0.1)
            no_ref_audio = gr.Checkbox(label="No Reference Audio", visible=False, value=False, scale=1, info="No mel condition")
            use_acc_grl = gr.Checkbox(label="Use accent grl condition", visible=False, value=True, scale=1, info="Use accent grl condition")
            use_prosody_encoder = gr.Checkbox(label="Use prosody encoder", visible=False, value=False, scale=1, info="Use prosody encoder")
            seed = gr.Number(label="Random Seed", scale=1, value=-1, minimum=-1)


    # Input fields
    ref_text = gr.Textbox(label="Reference Text", placeholder="Enter the text for the reference audio...")
    ref_audio = gr.Audio(label="Reference Audio", type="filepath", interactive=True, show_download_button=True, editable=True)


    with gr.Accordion("Denoise audio (Optional / Recommend)", open=True):
        with gr.Row():
            denoise_btn = gr.Button(value="Denoise")
            cancel_btn = gr.Button(value="Cancel Denoise")
        # Use numpy type here so we can reuse the waveform directly in Python.
        denoise_audio = gr.Audio(
            label="Denoised Audio",
            value=None,
            type="numpy",
            interactive=True,
            show_download_button=True,
            editable=True,
        )

    gen_text = gr.Textbox(label="Text to Generate", placeholder="Enter the text you want to generate...")

    # Inference button and outputs
    with gr.Row():
        txt_info_gpu = gr.Textbox("", label="Device Info")
        seed_info = gr.Textbox(label="Used Random Seed")
        check_button_infer = gr.Button("Generate Audio", variant="primary")

    gen_audio = gr.Audio(label="Generated Audio", type="filepath", interactive=True, show_download_button=True, editable=True)

    # Examples
    def _resolve_example(name: str) -> str:
        local = Path(PRETRAINED_ROOT)/ "demos" / "lemas_tts_test" / name
        return str(local) if local.is_file() else ""

    examples = gr.Examples(
        examples=[
            ["Te voy a dar un tip #1 que le copia a John Rockefeller, uno de los empresarios más picudos de la historia.",
            _resolve_example("es.wav"),
            "我要给你一个从历史上最精明的商人之一约翰·洛克菲勒那里抄来的秘诀。",
            ],
            ["Nova, #1 dia 25 desse mês vai rolar operação the last Frontier.",
            _resolve_example("pt.wav"),
            " Preparations are currently underway to ensure the operation proceeds as planned.",
            ],
        ],
        inputs=[
            ref_text,
            ref_audio,
            gen_text,
        ],
        outputs=[gen_audio, txt_info_gpu, seed_info],
        fn=infer,
        cache_examples=False
    )

    # System Info section at the bottom
    gr.Markdown("---")
    gr.Markdown("## System Information")
    with gr.Accordion("Update System Stats", open=False):
        update_button = gr.Button("Update System Stats", scale=1)
        output_box = gr.Textbox(label="GPU and CPU Information", lines=5, scale=5)

    def update_stats():
        return get_combined_stats()
        
    
    denoise_btn.click(fn=denoise,
                        inputs=[ref_audio],
                        outputs=[denoise_audio])

    cancel_btn.click(fn=cancel_denoise,
                        inputs=[ref_audio],
                        outputs=[denoise_audio])

    # Event handlers
    check_button_infer.click(
        fn=infer,
        inputs=[
            cm_project,
            cm_checkpoint,
            exp_name,
            ref_text,
            ref_audio,
            denoise_audio,
            gen_text,
            nfe_step,
            ch_use_ema,
            separate_langs,
            frontend,
            speed,
            cfg_strength,
            use_acc_grl,
            ref_ratio,
            no_ref_audio,
            sway_sampling_coef,
            use_prosody_encoder,
            seed,
        ],
        outputs=[gen_audio, txt_info_gpu, seed_info],
    )

    bt_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])
    cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])

    ref_audio.change(
            fn=lambda x: None,
            inputs=[ref_audio],
            outputs=[denoise_audio]
        )

    update_button.click(fn=update_stats, outputs=output_box)
    
    # Auto-load system stats on startup
    app.load(fn=update_stats, outputs=output_box)


@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print("Starting LEMAS-TTS Inference Interface...")
    print(f"Device: {device}")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        allowed_paths=[
            str(Path(PRETRAINED_ROOT) / "data"),
            str(Path(PRETRAINED_ROOT) / "demos"),
        ],
    )


if __name__ == "__main__":
    main()
