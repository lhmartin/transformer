from trainer import Trainer
from transformer import Transformer

## This entry allows for easier saving and loading of checkpoints.

if __name__ == '__main__':

    cfg = Trainer.Config(
        mdl_config=Transformer.Config(
            max_sequence_len=128,
            num_decoder_blocks=6,
            num_encoder_blocks=6,
            num_heads=8,
            ),
        batch_size=64,
        learing_rate=0.5,
        device='cuda',
        translation_dir='de_to_en'
    )

    trainer = Trainer(cfg)

    trainer.train()