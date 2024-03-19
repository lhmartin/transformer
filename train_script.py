from trainer import Trainer
from transformer import Transformer

## This entry allows for easier saving and loading of checkpoints.

if __name__ == '__main__':

    cfg = Trainer.Config(
        mdl_config=Transformer.Config(
            max_sequence_len=256,
            num_decoder_blocks=6,
            num_encoder_blocks=6,
            num_heads=8,
            ),
        batch_size=16,
        learing_rate=0.5,
        val_epoch_freq=20000,
        device='cuda',
        translation_dir='de_to_en'
    )

    trainer = Trainer(cfg)

    trainer.train()