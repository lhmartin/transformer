import wandb
from data.wmt14_dataset import WMT14_Dataset, DATASET_SPLITS
from pytorch_transformer import TransformerModel
from transformer import Transformer
from modules.other_model import Transformer as TransformerOther

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import no_grad, argmax
from pydantic import BaseModel
from tqdm import tqdm
from math import pow

class Trainer():

    class Config(BaseModel):
        learing_rate : float = 2.0
        batch_size : int = 256
        num_epochs : int = 1
        device     : str = 'cuda'
        mdl_config : Transformer.Config

    def __init__(self, config : Config) -> None:
        self._config = config

    def _instantiate_model(self):
        self.model = Transformer(self._config.mdl_config).to(self._config.device)
        self.model_other = TransformerOther(
            model_dimension=self._config.mdl_config.model_dimension,
            src_vocab_size=self._config.mdl_config.src_vocab_size,
            trg_vocab_size=self._config.mdl_config.tgt_vocab_size,
            number_of_heads=self._config.mdl_config.num_heads,
            number_of_layers=self._config.mdl_config.num_encoder_blocks,
            dropout_probability=0.1
        ).to(self._config.device)
        self.model.train(mode=True)
        self.model.init_params()

    def _create_optimizer(self):
        return AdamW(self.model_other.parameters(),
                    betas=(0.9, 0.98),
                    weight_decay=10e-9,
                    lr = self._config.learing_rate)

    def _create_scheduler(self, optimizer):

        def _schedule(step : int):

            # to account for step = 0
            step += 1

            dm = self._config.mdl_config.model_dimension
            warmup_steps = 4000

            lr =  pow(dm, -0.5) * min(pow(step, -0.5), step * pow(warmup_steps, -1.5))

            return lr

        return LambdaLR(optimizer, lambda x : _schedule(x))

    def _create_dataset(self, split : DATASET_SPLITS):
        return WMT14_Dataset(split=split)

    def _create_dataloader(self, split, shuffle=True) -> DataLoader:

        data = self._create_dataset(split)
        return DataLoader(dataset=data,
                          batch_size=self._config.batch_size,
                          collate_fn=self.model.collate_fn,
                          shuffle=shuffle,
                          )

    def _get_loss_fn(self) -> CrossEntropyLoss:
        return CrossEntropyLoss(label_smoothing=0.1,
                                ignore_index=self.model.tokenizer_de.pad_token_id,
                                reduction='mean')

    def train(self):

        run = wandb.init(
            project='transformer-testing',
            config = self._config.dict(),
        )

        self._instantiate_model()
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        loss_fn   = self._get_loss_fn()

        train_dataloader = self._create_dataloader('train', shuffle=True)
        val_dataloader = self._create_dataloader('validation', shuffle=False)

        for epoch_num in range(0, self._config.num_epochs):
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                optimizer.zero_grad()
                labels = batch['de']['input_ids']

                predictions = self.model_other(
                    src_token_ids_batch = batch['en']['input_ids'],
                    trg_token_ids_batch = labels,
                    src_mask=None, trg_mask=None)

                loss = loss_fn(
                    # flatten out the predictions and labels
                    predictions.contiguous().view(-1, 37000),
                    batch['de']['input_ids'].contiguous().view(-1)
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                if i % 1 == 0:
                    wandb.log({'train_loss' : loss})
                #     ids = argmax(predictions, dim=1)
                #     acc = (ids == labels)[labels != 0].float().sum() / labels[labels != 0].float().sum()
                #     wandb.log({'train_acc' : acc})

            print(f'Epoch {epoch_num} complete')

            val_loss = []
            with no_grad():
                for val_batch in val_dataloader:

                    predictions = self.model(
                        input_tkns = val_batch['en']['input_ids'],
                        target_tkns = val_batch['de']['input_ids']
                        )

                    loss = loss_fn(
                        # flatten out the predictions and labels
                        predictions.contiguous().view(-1, 37000),
                        val_batch['de']['input_ids'].contiguous().view(-1)
                    )
                    val_loss.append(loss.cpu())

            wandb.log({'val_loss' : sum(val_loss)/len(val_loss)})


if __name__ == '__main__':

    cfg = Trainer.Config(
        mdl_config=Transformer.Config(
            max_sequence_len=128,
            num_decoder_blocks=2,
            num_encoder_blocks=2,
            num_heads=4,
            ),
        batch_size=64,
        learing_rate=2.0,
    )

    trainer = Trainer(cfg)

    trainer.train()