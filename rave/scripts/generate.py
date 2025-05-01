import os, re
import tqdm
import torchaudio
import cached_conv as cc
from functools import partial
import torch, gin.torch
try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave
from absl import flags, app, logging
from itertools import product
from pathlib import Path

valid_exts = [".wav", ".aif", ".aiff", ".opus", ".mp3", ".aac"]
flags.DEFINE_multi_string('model', required=True, default=None, help="model path")
flags.DEFINE_multi_string('input', required=True, default=None, help="model inputs (file or folder)")
flags.DEFINE_multi_enum('mode', 'stream', ['stream', 'full'], help="streaming mode")
flags.DEFINE_multi_float('loudness', default = 0., help="loudness correction of incoming audio")
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')
flags.DEFINE_integer('chunk_size', default=8388608, help="chunk size for encoding/decoding (default: full file)")
flags.DEFINE_string('out_path', default='generations', help="output path")
FLAGS = flags.FLAGS

def process_stream(x, model, device, chunk_size=None):
    if chunk_size:
        x = list(x.split(chunk_size, dim=-1))
        if x[-1].shape[0] < chunk_size:
            x[-1] = torch.nn.functional.pad(x[-1], (0, FLAGS.chunk_size - x[-1].shape[-1]))
        x = torch.stack(x, 0)
    else:
        x = x[None]
    
    # forward into model
    out = []
    with torch.no_grad():
        for x_chunk in x:
            x_chunk_tmp = x_chunk.to(device)
            out_tmp = model(x_chunk_tmp[None])
            out.append(out_tmp.to('cpu'))
            del out_tmp, x_chunk_tmp
            torch.cuda.empty_cache()
    out = torch.cat(out, -1)
    return out

def process_full(x, model, device):
    x = x.to(device)
    out = model.forward(x[None])
    return out

def process_audio(path, mode, loudness, model=None, out_path="caca/", device=torch.device('cpu'), model_path=None, **kwargs):
    filepath, filetarget = path
    try:
        x, sr = torchaudio.load(filepath)
    except: 
        logging.warning('could not open file %s.'%filepath)
        return

    # load file
    if not hasattr(model, "sr"): model.sr = 48000 # for models trained with v1
    if not hasattr(model, "n_channels"): model.n_channels = 1 
    if sr != model.sr:
        x = torchaudio.functional.resample(x, sr, model.sr)
    if model.n_channels != x.shape[0]:
        if model.n_channels < x.shape[0]:
            x = x[:model.n_channels]
        else:
            print('[Warning] file %s has %d channels, butt model has %d channels ; skipping'%(f, model.n_channels))

    x = x * (pow(10,(loudness/20)))

    # process file
    if mode == "stream":
        out = process_stream(x, model, device, **kwargs)
    elif mode == "full":
        out = process_full(x, model, device)
    else:
        raise Exception("mode %s not known"%mode)

    # save file
    filename = Path(filetarget)
    target_path = Path(out_path)
    if model_path is not None:
        model_name = Path(model_path).stem
        target_path = target_path / model_name
    target_path = target_path / mode / f"{filename.stem}_loudness={loudness}.wav"
    os.makedirs(target_path.parent, exist_ok=True)
    torchaudio.save(str(target_path), out[0].cpu(), sample_rate=model.sr)

def load_model(model_path, device=torch.device('cpu')):
    if not os.path.exists(model_path):
        logging.error('path %s does not seem to exist.'%model_path)
        exit()
    if os.path.splitext(model_path)[1] == ".ts":
        model = torch.jit.load(model_path)
    else:
        config_path = rave.core.search_for_config(model_path)
        if config_path is None:
            logging.error('config not found in folder %s'%model_path)
        gin.parse_config_file(config_path)
        model = rave.RAVE()
        run = rave.core.search_for_run(model_path)
        if run is None:
            logging.error("run not found in folder %s"%model_path)
        model = model.load_from_checkpoint(run)
    return model.to(device)

def parse_audios(path):
    audio_files = []
    for root, _, files in os.walk(path):
        valid_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in valid_exts, files))
        audio_files.extend([(os.path.join(root, f), os.path.join(re.sub(path, '', root), f)) for f in valid_files])
    return audio_files

def main(argv):
    audio_files = sum(list(map(parse_audios, FLAGS.input)), [])
    cc.MAX_BATCH_SIZE = 8
    device = torch.device('cuda:%d'%FLAGS.gpu) if FLAGS.gpu >= 0 else torch.device('cpu')
    processes = []
    for model_path in FLAGS.model:
        logging.info('processing model %s'%model_path)
        model = load_model(model_path, device)
        with torch.no_grad():
            receptive_field = rave.core.get_minimum_size(model)
        if FLAGS.chunk_size:
            if FLAGS.chunk_size < receptive_field:
                logging.warning("chunk_size must be higher than %s receptive field (here : %s)"%(model_path, receptive_field))
                continue
        for kwargs in product(audio_files, FLAGS.mode, FLAGS.loudness):
            processes.append(partial(process_audio, *kwargs, device=device, model=model, out_path=FLAGS.out_path, chunk_size=FLAGS.chunk_size, model_path=model_path))
    for p in tqdm.tqdm(processes, desc="processing files for model %s..."%model_path):
        p()

if __name__ == "__main__":
    app.run(main)
    
