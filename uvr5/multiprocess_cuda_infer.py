import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from random import shuffle
from tqdm import tqdm
import sys, wave
import torch, torchaudio
import hashlib
import time, os, psutil

# Make sure we resolve imports relative to this bundled copy of uvr5
THIS_FILE = os.path.abspath(__file__)
UVR5_ROOT = os.path.dirname(THIS_FILE)
if UVR5_ROOT not in sys.path:
    sys.path.append(UVR5_ROOT)

from gui_data.constants import *
from lib_v5.vr_network.model_param_init import ModelParameters
import argparse, json
import onnx
import onnxruntime as ort
import traceback
from datetime import datetime

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

class ModelData():
    def __init__(self, 
                    model_path,
                    audio_path,
                    result_path,
                    process_method,
                    device, 
                    save_background=True,
                    is_pre_proc_model=False,
                    base_dir=UVR5_ROOT, 
                    **kwargs):
        self.__dict__.update(kwargs)

        BASE_PATH = result_path
        VR_MODELS_DIR = os.path.join(base_dir, 'models', 'VR_Models')
        VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
        VR_PARAM_DIR = os.path.join(base_dir, 'lib_v5', 'vr_network', 'modelparams')
        SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')

        MDX_MIXER_PATH = os.path.join(base_dir, 'lib_v5', 'mixer.ckpt')
        # MDX_MODELS_DIR = os.path.join(base_dir, 'models', 'MDX_Net_Models')
        # MDX_HASH_DIR = (base_dir, 'models', 'MDX_Net_Models', 'model_data')
        MDX_HASH_JSON = os.path.join(base_dir, 'model_data.json')
        MDX_MODEL_NAME_SELECT = os.path.join(base_dir, 'model_name_mapper.json')

        self.model_name = self.model_name
        self.aggression_setting = float(int(self.aggression_setting)/100) # 1 - 20
        self.window_size = int(self.window_size)
        self.batch_size = int(self.batch_size) if self.batch_size.isdigit() else 1
        self.mdx_batch_size = 1 if self.mdx_batch_size == DEF_OPT else int(self.mdx_batch_size)
        self.is_mdx_ckpt = False
        self.crop_size = int(self.crop_size) 
        self.is_high_end_process = 'mirroring' if self.is_high_end_process else 'None'
        self.post_process_threshold = float(self.post_process_threshold)
        self.model_capacity = 32, 128
        self.model_path = model_path
        self.result_path = result_path
        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        self.mixer_path = MDX_MIXER_PATH
        self.process_method = process_method
        self.is_pre_proc_model = is_pre_proc_model
        self.vr_is_secondary_model = self.vr_is_secondary_model_activate
        self.mdx_is_secondary_model = self.mdx_is_secondary_model_activate
        self.is_ensemble_mode = False
        self.secondary_model = None
        self.primary_model_primary_stem = None
        self.primary_stem = None
        self.secondary_stem = None
        self.secondary_model_scale = None
        self.is_demucs_pre_proc_model_inst_mix = False
        self.device = device
        self.save_background = save_background

        if type(audio_path)==str and os.path.isdir(audio_path):
            self.inputPaths = os.listdir(audio_path)
            self.inputPaths = [os.path.join(audio_path, x) for x in self.inputPaths if x[-4:]=='.wav']
        elif type(audio_path)==str and audio_path[-4:] == '.wav':
            self.inputPaths = [audio_path]
        elif type(audio_path) == list and audio_path[0][-4:] == '.wav':
            self.inputPaths = audio_path
        else:
            print(f"Invalid audio_path {audio_path}")

        self.get_model_hash()
        
        if self.process_method == VR_ARCH_TYPE:
            self.model_data = json.loads(open(VR_HASH_JSON, 'r', encoding='utf-8').read())[self.model_hash]
            if self.model_data:
                vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                self.primary_stem = self.model_data["primary_stem"]
                self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                self.vr_model_param = ModelParameters(vr_model_param)
                self.model_samplerate = self.vr_model_param.param['sr']
                if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                    self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                    self.is_vr_51_model = True
            else:
                self.model_status = False
        
        
        if self.process_method == MDX_ARCH_TYPE:
            self.is_vr_51_model = False
            self.margin = int(self.margin)
            self.model_samplerate = self.margin
            self.chunks = self.determine_auto_chunks(self.chunks) if self.is_chunk_mdxnet else 0
            self.model_data = json.loads(open(MDX_HASH_JSON, 'r', encoding='utf-8').read())[self.model_hash]
            if self.model_data:
                self.is_secondary_model = self.mdx_is_secondary_model
                self.compensate = self.model_data["compensate"]
                self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                self.primary_stem = self.model_data["primary_stem"]
                self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
            else:
                self.model_status = False


    def determine_auto_chunks(self, chunks):
        """Determines appropriate chunk size based on user computer specs"""
        gpu = 0 if torch.cuda.device_count() > 0 else -1
        if OPERATING_SYSTEM == 'Darwin':
            gpu = -1

        if chunks == BATCH_MODE:
            chunks = 0
            #self.chunks_var.set(AUTO_SELECT)

        if chunks == 'Full':
            chunk_set = 0
        elif chunks == 'Auto':
            if gpu == 0:
                gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
                if gpu_mem <= int(6):
                    chunk_set = int(5)
                if gpu_mem in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    chunk_set = int(10)
                if gpu_mem >= int(16):
                    chunk_set = int(40)
            if gpu == -1:
                sys_mem = psutil.virtual_memory().total >> 30
                if sys_mem <= int(4):
                    chunk_set = int(1)
                if sys_mem in [5, 6, 7, 8]:
                    chunk_set = int(10)
                if sys_mem in [9, 10, 11, 12, 13, 14, 15, 16]:
                    chunk_set = int(25)
                if sys_mem >= int(17):
                    chunk_set = int(60) 
        elif chunks == '0':
            chunk_set = 0
        else:
            chunk_set = int(chunks)
        print("chunks: ", gpu_mem, chunk_set)
        return chunk_set


    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()


class Inference():
    def __init__(self, model_data: ModelData, device):
        self.device = device
        self.n_fft = model_data.mdx_n_fft_scale_set
        self.is_normalization = model_data.is_normalization
        self.compensate = model_data.compensate
        self.dim_f, self.dim_t = model_data.mdx_dim_f_set, 2**model_data.mdx_dim_t_set
        self.mdx_batch_size = model_data.mdx_batch_size
        self.is_denoise = model_data.is_denoise
        self.hop = 1024
        self.dim_c = 4
        self.chunks = model_data.chunks
        self.margin = model_data.margin
        self.adjust = 1
        self.progress_value = 0

        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.chunk_size = self.hop * (self.dim_t-1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=False).to(self.device)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins-self.dim_f, self.dim_t]).to(self.device)
        self.gen_size = self.chunk_size-2*self.trim
        self.save_background = model_data.save_background


    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True,return_complex=True)
        x=torch.view_as_real(x)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,self.dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x=x.contiguous()
        x=torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,2,self.chunk_size])


    def load_model(self, model_path, threads, device="cuda"):
        model = onnx.load_model(model_path)
        if torch.cuda.is_available() and device != "cpu":
            providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        # sess_options.enable_profiling = True # debug 时开启
        self.ort_ = ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=providers)

        self.model_run = lambda spek:self.ort_.run(None, {'input': spek.cpu().numpy()})[0]


    def initialize_mix(self, mix):
        mix_waves = []
        n_sample = mix.shape[1]
        pad = self.gen_size - n_sample%self.gen_size
        zero_pad = torch.zeros((2,self.trim), device=mix.device)
        # print("mix:", mix.shape, mix.device, "zero_pad:", zero_pad.shape, zero_pad.device)
        mix_p = torch.cat((zero_pad, mix, torch.zeros((2,pad), device=mix.device), zero_pad), 1)
        i = 0
        while i < n_sample + pad:
            waves = mix_p[:, i:i+self.chunk_size]
            mix_waves.append(waves.unsqueeze(0))
            i += self.gen_size
            # print("debug 7:", i, waves, waves.shape, self.gen_size)
        mix_waves = torch.cat(mix_waves, 0).to(self.device)
        # print("debug 8:", mix_waves, mix_waves.shape, self.device, pad)
        return mix_waves, pad


    def run_model(self, mix, is_match_mix=False):
        
        spek = self.stft(mix.to(self.device))*self.adjust
        spek[:, :, :3, :] *= 0 
        # print("spek input:", spek.device, spek.shape)
        if is_match_mix:
            spec_pred = spek.to(self.device)
        else:
            spec_pred = -self.model_run(-spek)*0.5+self.model_run(spek)*0.5 if self.is_denoise else self.model_run(spek)
            spec_pred = torch.from_numpy(spec_pred).to(self.device)

        # print("is_denoise:", self.is_denoise, "spec_pred:", spec_pred.dtype, type(spec_pred))
        return self.istft(spec_pred).to(self.device)[:,:,self.trim:-self.trim].transpose(0,1).reshape(2, -1)


    def demix_base(self, mix, is_match_mix=False, device='cpu'):
        chunked_sources = []
        
        for slice in mix:
            # print("debug 6:", mix, slice, is_match_mix)
            sources = []
            tar_waves_ = []
            mix_p = mix[slice]
            # print("demix_base: ", mix_p.shape, mix_p.device)
            mix_waves, pad = self.initialize_mix(mix_p.to(device))
            mix_waves = mix_waves.split(self.mdx_batch_size)
            with torch.no_grad():
                for mix_wave in mix_waves:
                    # self.running_inference_progress_bar(len(mix)*len(mix_waves), is_match_mix=is_match_mix)
                    # print("debug10:", mix_wave, mix_wave.shape, is_match_mix)
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)
                    tar_waves_.append(tar_waves)

                tar_waves = torch.cat(tar_waves_, axis=-1)[:, :-pad]
                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
                sources.append(tar_waves[:,start:end]*(1/self.adjust))
            chunked_sources = torch.cat(sources, axis=-1)
        # print("debug 11:",chunked_sources, len(chunked_sources), chunked_sources.shape)
        # sources = torch.cat(chunked_sources, axis=-1)
        sources = chunked_sources
        # print("debug 4:", sources, sources.shape)
        return sources

    def onnx_inference(self, wav_path, save_dir, device):
        start_time = time.time()
        input_audio, sr = torchaudio.load(wav_path, channels_first=True)
        input_audio = input_audio.to(device)
        # input_audio = input_audio.mean(dim=0).unsqueeze(0)  # stereo to mono
        if input_audio.shape[0] == 1:
            input_audio = torch.cat((input_audio, input_audio), 0) # mono to stereo
        if sr != 44100:
            input_audio = torchaudio.functional.resample(input_audio.squeeze(), sr, 44100)

        output_audio = self.demix_base({0:input_audio.squeeze()}, is_match_mix=False, device=device)
        torchaudio.save(
            os.path.join(save_dir, os.path.basename(wav_path).replace(".wav", "_vocal.wav")),
            output_audio.cpu(),
            44100,
        )

        if self.save_background:
            raw_mix = self.demix_base({0:input_audio.squeeze()}, is_match_mix=True)
            secondary_source, raw_mix = normalize_two_stem(output_audio*self.compensate, raw_mix, self.is_normalization)
            secondary_source = (-secondary_source+raw_mix)
            torchaudio.save(
                os.path.join(save_dir, os.path.basename(wav_path)).replace(".wav", "_background.wav"),
                secondary_source.cpu(),
                44100,
            )
        process_time = time.time() - start_time
        print(f"{datetime.now()} {wav_path} denoised time: {process_time:.3f}s audio len: {output_audio.shape[-1]/44100:.3f}s RTF: {output_audio.shape[-1]/44100/process_time:.3f}")
        
        vocal_path = os.path.join(save_dir, os.path.basename(wav_path).replace(".wav", "_vocal.wav"))
        bg_path = os.path.join(save_dir, os.path.basename(wav_path).replace(".wav", "_background.wav")) if self.save_background else ""
        
        return vocal_path, bg_path

def normalize_two_stem(wave, mix, is_normalize=False):
    """Save output music files"""
    
    maxv = torch.abs(wave).max()
    max_mix = torch.abs(mix).max()
    
    if maxv > 1.0:
        # print(f"\nNormalization Set {is_normalize}: Primary source above threshold for clipping. Max:{maxv}")
        # print(f"\nNormalization Set {is_normalize}: Mixture above threshold for clipping. Max:{max_mix}")
        if is_normalize:
            wave /= maxv
            mix /= maxv
    
    return wave, mix  


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # 获取音频帧数
        n_frames = wav_file.getnframes()
        # 获取采样率
        framerate = wav_file.getframerate()
        # 计算时长（秒）
        duration = n_frames / float(framerate)
    return duration


def walkFile(data_dir, save_dir):
    res_wavs = []
    res_txts = []
    for root, dirs, files in tqdm(os.walk(data_dir)):
        # 遍历文件
        for f in files:
            if f[-4:] == '.wav':
                wav_path = os.path.join(root, f)
                if not os.path.exists(os.path.join(save_dir, f'{f[:-4]}_Vocals.wav')):
                    res_wavs.append(wav_path)
            # elif f[-4:] == '.csv':
            #     res_txts.append(os.path.join(root, f))
            
    return res_wavs # , res_txts


def process_batch(files, args, device='cpu'):

    configs = json.loads(open(args.config_path, 'r', encoding='utf-8').read())
    model_data = ModelData(
        model_path=args.model_path,
        audio_path = files,
        result_path = args.result_path,
        process_method = args.process_method,
        device = device,
        save_background = args.save_background,
        **configs
    )
    # uvr5_model = Inference_raw(model_data, device)
    # uvr5_model.process_start()

    uvr5_model = Inference(model_data, device)
    uvr5_model.load_model(args.model_path, args.num_processes)
    print(f"Loaded UVR5 model in {device}.")

    for file in files:
        vocal_path, bg_path = uvr5_model.onnx_inference(file, os.path.join(args.result_path, os.path.basename(file)), device)



def parallel_process(filenames, args):
    total_gpu = torch.cuda.device_count()
    print(f'Total GPUs: {total_gpu}')
    with ProcessPoolExecutor(max_workers=args.num_processes*total_gpu) as executor:
        tasks = []
        for i in range(args.num_processes):
            start = int(i * len(filenames) / args.num_processes)
            end = int((i + 1) * len(filenames) / args.num_processes)
            file_chunk = filenames[start:end]
            for n in range(total_gpu):
                chunk = file_chunk[int(n*len(file_chunk)/total_gpu): int((n+1)*len(file_chunk)/total_gpu)]
                device = f"cuda:{n}" if torch.cuda.is_available() else "cpu"
                print("load model in devices: ", args.num_processes, total_gpu, i, n, device)
                tasks.append(executor.submit(process_batch, chunk, args, device))

        for task in tqdm(tasks):
            task.result()


def parallel_process_cpu(filenames, args):
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        tasks = []
        for i in range(args.num_processes):
            start = int(i * len(filenames) / args.num_processes)
            end = int((i + 1) * len(filenames) / args.num_processes)
            chunk = filenames[start:end]
            print("load model in devices: ", args.num_processes, i, "cpu")
            tasks.append(executor.submit(process_batch, chunk, args, "cpu"))
        for task in tqdm(tasks):
            task.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str, default="models/MDX_Net_Models/model_data/Kim_Vocal_1.onnx", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="models/MDX_Net_Models/model_data/MDX-Net-Kim-Vocal1.json", help='配置文件路径') 
    parser.add_argument('-a', '--audio_path', type=str, default="", help='wav文件名列表，放在raw文件夹下')
    parser.add_argument('-r', '--result_path', type=str, default="", help='结果存储路径')
    parser.add_argument('-p', '--process_method', type=str, default="MDX-Net", help='可选方法:["VR Arc", "MDX-Net"]')
    parser.add_argument('-b', '--save_background', type=bool, default=True, help='True:保存人声和背景音，False:只保存人声')
    parser.add_argument('-w', '--num_processes', type=int, default=4, help='You are advised to set the number of processes to the same as the number of CPU cores')

    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)
    if args.save_background:
        os.makedirs(os.path.join(os.path.dirname(args.result_path), "bg_music"), exist_ok=True)

    if os.path.isdir(args.audio_path):
        filenames = walkFile(args.audio_path, args.result_path)
    elif args.audio_path.endswith(".wav"):
        filenames = [args.audio_path]

    # shuffle(filenames)
    print(len(filenames))

    # process_batch(filenames, args, "cpu")

    multiprocessing.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()

    if torch.cuda.is_available():
        parallel_process(filenames, args)
    else:
        parallel_process_cpu(filenames, args)
