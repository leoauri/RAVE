from pathlib import Path
import shutil
import os
import sys

import gin
import torch
from absl import flags, app, logging

import acids_dataset as ad
from acids_dataset import transforms as adt, features as adf
from acids_dataset.features import ModuleEmbedding

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

try: 
    from after.diffusion import RectifiedFlow, EDM
    from after.diffusion.utils import collate_fn
except: 
    logging.error("Could not find AFTER dependency. Exiting...")
    exit()

import rave.dataset


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def parse_flags():

    flags.DEFINE_string('name', None, help='Name of the run')
    flags.DEFINE_string('model', default=None, required=True, help="pretrained RAVE path")
    flags.DEFINE_string('config', default="base", help="AFTER model config (rectified or edm)")
    flags.DEFINE_string('arch', default="rectified", help="AFTER model config (rectified or edm)")
    flags.DEFINE_string('out_path', default="runs/", help="out directory path")
    flags.DEFINE_string('embedding_name', default=None, help="sets embedding name (default : RAVE checkpoint name)")
    flags.DEFINE_bool('force', default=False, help="Forces re-processing of augmentations & features")

    # Augmentation-related keywords 
    flags.DEFINE_bool('force_augmentation', default=False, help="Forces re-processing of augmentations")
    flags.DEFINE_integer('n_augmentations', default=4, help="number of data augmentations for AFTER training (default: 4)")
    flags.DEFINE_bool('stretch', default=True, help="augments data with time stretching (default : True)")
    flags.DEFINE_list('stretch_range', default='0.7,1.6', help="time-stretch range (default: 0.7,1.6)")
    flags.DEFINE_bool('shift', default=True, help="augments data with pitch-shifting (default : True)")
    flags.DEFINE_list('shift_range', default='-6,6', help="pitch shift range (default: -6,6)")

    # Feature-related keywords
    flags.DEFINE_bool('force_features', default=False, help="Forces re-processing of features")
    flags.DEFINE_bool('structure', default=None, help="Information from strcture (midi or beat).")
    flags.DEFINE_bool('midi', default=False, help="enables MIDI extraction from MIDI files (default: False)")
    flags.DEFINE_bool('beat', default=False, help="enables beat extraction from audio files (default: False)")

    # Data-related key words
    flags.DEFINE_multi_string('db_path', default=None, required=True, help="Preprocessed dataset paths")
    flags.DEFINE_multi_float('db_weights', default=None, required=False, help="Weights for dataset balancing")
    flags.DEFINE_multi_float('db_partition', default=None, help="Fetches given partition in dataset(s)")
    flags.DEFINE_integer('db_max_size', default=None, help="Fetches given partition in dataset(s)")
    flags.DEFINE_boolean('use_cache_for_multi', default=True, required=False, help="Use cache for multi-dataset random sampling (only for multi dataset training)")

    # Training related keywords
    flags.DEFINE_string('device', default="cpu", help='device to use (default : cpu, can be set to cuda:0, cuda:1, mps...)')
    flags.DEFINE_integer('batch', 32, help="batch size")
    flags.DEFINE_integer('n_signal', None, help="chunk size (default: chunk_size // 2)")
    flags.DEFINE_string('ckpt', default=None, help="checkpoint to resume")
    flags.DEFINE_float('fidelity', default=0.99, help="prior target fidelity")
    flags.DEFINE_integer('workers',
                        default=8,
                        help='Number of workers to spawn for dataset loading')
    flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
    flags.DEFINE_integer('save_every',
                        None,
                        help='save every n steps (default: just last)')
    flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
    flags.DEFINE_bool('derivative',
                    default=False,
                    help='Train RAVE on the derivative of the signal')
    flags.DEFINE_bool('normalize',
                    default=False,
                    help='Train RAVE on normalized signals')
    flags.DEFINE_bool('smoke_test', 
                    default=False, 
                    help="runs in smoke test (no training)")



def main(argv):

    FLAGS = flags.FLAGS
    device = torch.device(FLAGS.device)
    if FLAGS.structure == "midi":
        logging.info('structure with MIDI only works with midi architecture ; changing')
        FLAGS.config = "midi"      
    if FLAGS.config == "midi" and FLAGS.structure != "midi": 
        logging.info('structure with MIDI only works with midi architecture ; changing')
        FLAGS.structure = "midi"


    # [First step] initialize model model (either scripted or checkpoint)
    logging.info("Initializing model %s"%FLAGS.model)
    assert os.path.exists(FLAGS.model), f"{FLAGS.model} not found"
    if os.path.splitext(FLAGS.model)[1] == ".ts":
        rave_model = torch.jit.load(FLAGS.model)
        embedding_name = FLAGS.embedding_name or os.path.splitext(os.path.basename(FLAGS.model))
        z_downsample = rave_model.encode_params[1]
        z_shape = rave_model.encode_params[2]
        method_kwargs = {}
        logging.info(f"Model found at {FLAGS.model}, dim={z_shape}, ratio={z_downsample}")
    else: 
        rave_model, run_path = rave.load_rave_checkpoint(FLAGS.model)
        embedding_name = FLAGS.embedding_name or rave.get_run_name(run_path)
        logging.info('model found : %s'%run_path)
        rave_sr = rave_model.sr
        z_shape = rave_model.latent_size_from_fidelity(FLAGS.fidelity)
        # compute z downsampling
        z = rave_model.encode(torch.randn(1, 1, 4096))
        z_downsample = 4096 // z.shape[-1]
        method_kwargs = {'fidelity': FLAGS.fidelity}
        logging.info(f"Model found at {run_path}, dim={z_shape}, ratio={z_downsample}")

    #
    # [Second step] init augmentation transforms
    logging.info("Initializing augmentation transforms...")
    rave_transforms = []
    if FLAGS.derivative: 
        rave_transforms += [adt.Derivator(sr=rave_model.sr, p=1.)]
    if FLAGS.normalize:
        rave_transforms += [adt.Normalize(p=1.)]
    augment_transforms = []
    if (FLAGS.stretch or FLAGS.shift):
        for i in range(FLAGS.n_augmentations): 
            pitch_transform = [adt.PitchShift(min_semitones=float(FLAGS.shift_range[0]), max_semitones=float(FLAGS.shift_range[1]), p=1.0, sr=rave_sr),
                                                 adt.TimeMask(min_band_part=0.07, max_band_part=0.15, fade=True, p=1.0, sr=rave_sr)]
            stretch_transform = rave_transforms + [adt.TimeStretch(min_rate=float(FLAGS.stretch_range[0]), max_rate=float(FLAGS.stretch_range[1]), sr=rave_sr),
                                                   adt.TimeMask(min_band_part=0.07, max_band_part=0.15, fade=True, p=1.0, sr=rave_sr)]
            if (FLAGS.stretch and FLAGS.shift): 
                augment_transforms.append([*rave_transforms, *stretch_transform, *pitch_transform])
            elif (FLAGS.stretch):
                augment_transforms.append([*rave_transforms, *stretch_transform])
            elif (FLAGS.shift):
                augment_transforms.append([*rave_transforms, *stretch_transform])
    embedding_feature = ModuleEmbedding(module=rave_model, module_sr=rave_model.sr, 
                                        method="encode_compressed", method_kwargs=method_kwargs, 
                                        name=embedding_name, transforms=augment_transforms, device=device) 
    logging.info("Resulting embedding transform : %s..."%embedding_feature)

    
    # [Third step] init structure information and corresponding features if needed
    features = []
    logging.info(f"Parsing features for structure {FLAGS.structure}...")
    time_transform = None

    if (FLAGS.midi or FLAGS.structure == "midi"):
        midi_transform = adf.AfterMIDI(device=device)
        features.append(midi_transform)
    if (FLAGS.beat or FLAGS.structure == "beat"):
        time_transform = adf.BeatTrack(downsample=z_downsample, device=device)
        features.append(time_transform)
    logging.info(f"Parsed features : {features}")

    
    # [Fourth step] Update database
    logging.info("Parsing and updating databases...")
    chunk_size = None
    for data_path in FLAGS.db_path:
        dataset_metadata = ad.get_metadata_from_path(data_path)
        dataset_sr = dataset_metadata['sr']
        dataset_chunk_size = int(dataset_metadata.get('chunk_length') * dataset_metadata.get('sr'))
        chunk_size = dataset_chunk_size if chunk_size is None else min(dataset_chunk_size, chunk_size)

        # check latent embedding
        force_recompute_aug = FLAGS.force or FLAGS.force_augmentation
        if (embedding_name in ad.get_feature_names_from_path(data_path)) and not force_recompute_aug:
            # check if existing transform matches the current training
            saved_embedding = ad.get_features_from_path(data_path)[embedding_name]
            assert saved_embedding.n_augmentations == FLAGS.n_augmentations, \
                "embedding %s exists in data path {data_path}, but has %d augmentations (asked: %d.)\n" \
                "you can force recomputation of the augmentations with the --force_augmentation feature"
            for i, t in enumerate(saved_embedding.transforms): 
                if type(t) != type(augment_transforms[i]): 
                    logging.warning(f"transform #{i} ({augment_transforms[i]}) does not match with loaded embedding ({t})")

        # update embedding
        logging.info(f"Parsing embeddings for {data_path}...")
        ad.update_dataset(data_path, features = [embedding_feature], overwrite=force_recompute_aug, max_db_size=FLAGS.db_max_size)

        # update features
        force_recompute_feat = FLAGS.force or FLAGS.force_features
        if len(features) > 0:
            logging.info(f"Parsing features for {data_path}...")
            ad.update_dataset(data_path, features = features, overwrite=force_recompute_feat)


    # [Fifth step] Init datasets
    logging.info(f"Initializing datasets...")
    aug_keys = [f"aug_{i}" for i in range(FLAGS.n_augmentations)]
    data_keys = [f"{embedding_name}[0]->z", 
                 *[f"{embedding_name}[{i+1}]->{aug_keys[i]}" for i in range(FLAGS.n_augmentations)]]
    if FLAGS.structure == "beat": 
        data_keys += ["waveform"]
    elif FLAGS.structure == "midi": 
        data_keys += [f"{midi_transform.feature_name}->midi"]

    output_format = "{" + ",".join(data_keys) + "}"
    gin.bind_parameter("diffusion.utils.collate_fn.timbre_augmentation_keys", aug_keys)
                              
    dataset = []
    for d in FLAGS.db_path: 
        dataset.append(ad.datasets.AudioDataset(d, output_pattern=output_format, required_fields=['waveform', embedding_name]))
    if len(dataset) == 1: 
        dataset = dataset[0]
    else:
        dataset = ad.datasets.CombinedAudioDataset(dataset, FLAGS.weights, max_samples=FLAGS.max_samples, use_cache=FLAGS.use_cache)
    logging.info(f"Parsed dataset total length : {len(dataset)}...")
    partitions = dataset.split({'train': 0.98, 'val': 0.02}, load_if_available=FLAGS.db_partition, use_meta_if_available=True)
    train_dataset = partitions['train']; val_dataset = partitions['val']
    logging.info(f"Train size : {len(train_dataset)}; validation size : {len(val_dataset)}")

    # check data loading
    try: 
        data = train_dataset[0]
        data = val_dataset[0]
    except Exception as e: 
        logging.error("Caught error when testing data fetching. Got : ")
        e.with_traceback()
   
    
    # [Sixth step] Init AFTER models
    logging.info(f"Initializing AFTER model")

    gin.clear_config()
    
    if FLAGS.ckpt is not None: 
        config_path = os.path.join(FLAGS.out_path, FLAGS.name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [FLAGS.override])
    else:
        gin.parse_config_files_and_bindings(
            [add_gin_extension(FLAGS.config)],
            FLAGS.override,
        )

    if not FLAGS.n_signal:
        FLAGS.n_signal = (chunk_size // 2)
    n_after_latents = FLAGS.n_signal // z_downsample
    logging.info("learned latent steps : %s"%(n_after_latents))
    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.collate_fn.ae_ratio", z_downsample)
        gin.bind_parameter("%IN_SIZE", z_shape)
        gin.bind_parameter("%N_SIGNAL", n_after_latents)

    if FLAGS.arch == "rectified":
        after_model = RectifiedFlow(device=device, emb_model=rave_model, time_transform=time_transform)
    elif FLAGS.arch == "edm":
        after_model = EDM(device=device, emb_model=rave_model, time_transform=time_transform)

    # [Seventh step] Init data loaders
    logging.info(f"Initializing training")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=FLAGS.batch, 
        sampler = train_dataset.get_sampler(), 
        num_workers=FLAGS.workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=FLAGS.batch, 
        shuffle=False,
        num_workers=FLAGS.workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # [Eight step] Prepare checkpoints & logging number of paramters
    dataset._loader[0].get_array('musicnet_4f3923fb55')
    logging.info(f"Preparing training...")
    model_dir = os.path.join(FLAGS.out_path, FLAGS.name)
    os.makedirs(model_dir, exist_ok=True)
    num_el = 0
    for p in after_model.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    if after_model.encoder is not None:
        num_el = sum([p.numel() for p in after_model.encoder.parameters()])
        print("Number of parameters - encoder : ", num_el / 1e6, "M")

    if after_model.encoder_time is not None:
        num_el = sum([p.numel() for p in after_model.encoder_time.parameters()])
        print("Number of parameters - encoder_time : ", num_el / 1e6, "M")

    if after_model.classifier is not None:
        num_el = sum([p.numel() for p in after_model.classifier.parameters()])
        print("Number of parameters - classifier : ", num_el / 1e6, "M")

    ######### TRAINING #########
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": val_loader,
        "restart_step": FLAGS.ckpt,
    }

    logging.info(f"Starting training...")
    if FLAGS.smoke_test: 
        logging.info('smoke_test succeeded!')
        pass
    else:
        after_model.fit(**d)



if __name__ == "__main__": 
    parse_flags() 
    app.run(main)
