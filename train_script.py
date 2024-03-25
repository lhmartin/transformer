from trainer import Trainer
from transformer import Transformer
from torch import load
from torch.cuda import device_count
from pydantic_yaml import parse_yaml_file_as
import argparse
import torch.multiprocessing as mp
## This entry allows for easier saving and loading of checkpoints.

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help='The location of the config file to use', dest='config_fp')

    args = argparser.parse_args()
    if args.config_fp:
        cfg = parse_yaml_file_as(Trainer.Config, args.config_fp)
    else:
        cfg = Trainer.Config(
            mdl_config=Transformer.Config(
                max_sequence_len=128,
                num_decoder_blocks=6,
                num_encoder_blocks=6,
                num_heads=8,
                ),
            batch_size=64,
            learing_rate=0.5,
            val_epoch_freq=1,
            device='gpu',
            gpus=[0,1],
            translation_dir='de_to_en',
            loss_fn='cross_entropy'
        )

    trainer = Trainer(cfg)
    ckpt = None

    if cfg.resume_from_ckpt is not None:
        ckpt = load(open(cfg.resume_from_ckpt, 'rb'))

    if trainer.multi_gpu_mode:
        world_size = device_count()
        mp.spawn(trainer.train, args=(world_size, ckpt), nprocs=world_size)
    else:
        trainer.train(ckpt=ckpt, rank=0, world_size=1)