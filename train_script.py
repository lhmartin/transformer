from trainer import Trainer
from transformer import Transformer
from torch import load
from pydantic_yaml import parse_yaml_file_as, to_yaml_str
import argparse
import yaml
## This entry allows for easier saving and loading of checkpoints.

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, help='The location of the config file to use', dest='config_fp')

    args = argparser.parse_args()
    if args.config_fp:
        cfg = parse_yaml_file_as(Trainer.Config, 'configs/base_transformer.yaml')
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
            val_epoch_freq=25000,
            device='cuda',
            translation_dir='de_to_en'
        )

    trainer = Trainer(cfg)
    ckpt = None

    if cfg.resume_from_ckpt is not None:
        ckpt = load(open(cfg.resume_from_ckpt, 'rb'))

    trainer.train(ckpt)