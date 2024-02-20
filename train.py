import wandb
from data.wmt14_dataset import WMT14_Dataset, DATASET_SPLITS
from transformer import Transformer

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import no_grad, argmax
from pydantic import BaseModel
from tqdm import tqdm
from math import sqrt, pow

class Trainer():

    class Config(BaseModel):
        learing_rate : float = 0.001
        batch_size : int = 16
        num_epochs : int = 100
        device     : str = 'cuda'
        mdl_config : Transformer.Config

    def __init__(self, config : Config) -> None:
        self._config = config

    def _instantiate_model(self):
        self.model = Transformer(self._config.mdl_config).to(self._config.device)

    def _create_optimizer(self):
        return Adam(self.model.parameters(),
                    betas=(0.9, 0.98),
                    weight_decay=10e-9,
                    lr = self._config.learing_rate,)

    def _create_scheduler(self, optimizer):

        def _schedule(step : int):

            # to account for step = 0
            step += 1

            dm = self._config.mdl_config.model_dimension
            warmup_steps = 4000

            lr =  pow(dm, -0.5) * min(pow(step, -0.5), step * pow(warmup_steps, -1.5))

            wandb.log({'Learning weight' : lr })

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
        return CrossEntropyLoss(label_smoothing=0.1,ignore_index=self.model.tokenizer_en.pad_token_id)

    def train(self):

        run = wandb.init(
            project='transformer-testing',
            config = self._config.dict(),
        )
        text_table = wandb.Table(columns=["step", "loss", "pred_text", "target_text"])

        self._instantiate_model()
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        loss_fn   = self._get_loss_fn()

        train_dataloader = self._create_dataloader('train', shuffle=True)
        val_dataloader = self._create_dataloader('validation', shuffle=False)

        for epoch_num in range(0, self._config.num_epochs):
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                optimizer.zero_grad()

                predictions = self.model(
                    input_tkns = batch['en']['input_ids'],
                    target_tkns = batch['de']['input_ids']
                    )

                loss = loss_fn(
                    # flatten out the predictions and labels
                    predictions.contiguous().view(-1, 37000),
                    batch['de']['input_ids'].contiguous().view(-1)
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                if i % 10 == 0:
                    pred_ids = argmax(predictions[0], dim=1)
                    decoded_pred = self.model.tokenizer_de.decode(pred_ids)
                    decoded_target = self.model.tokenizer_de.decode(batch['de']['input_ids'][0])

                    text_table.add_data(i, loss, decoded_pred, decoded_target)
                    run.log({'samples' : text_table})

                    wandb.log({'train_loss' : loss})


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
        mdl_config=Transformer.Config(),
        batch_size=16
    )

    trainer = Trainer(cfg)

    trainer.train()