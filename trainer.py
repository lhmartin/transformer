import wandb
from data.wmt14_dataset import WMT14_Dataset, DATASET_SPLITS
from transformer import Transformer

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import Tensor, no_grad, save
from pydantic import BaseModel
from tqdm import tqdm
from math import pow
import datetime

from utils.metrics import decode_and_calculate_bleu_score, calculate_accuracy

class Trainer():
    """
    Manages the training of the Transformer model
    """

    class Config(BaseModel):
        checkpoint_folder : str = './transformer_checkpoints/'
        checkpoint_steps  : int = 10000     # Number of steps to then save a checkpoint
        val_epoch_freq    : int = 10000     # How many steps have to be taken until running through the val dataloader
        learing_rate      : float = 2.0
        batch_size        : int = 256
        num_epochs        : int = 10
        device            : str = 'cuda'
        logging_freq      : int = 10
        warmup_steps      : int = 4000
        mdl_config        : Transformer.Config

    def __init__(self, config : Config) -> None:
        self._config = config
        self.checkpoint_folder = self._config.checkpoint_folder + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

    def _instantiate_model(self):
        self.model = Transformer(self._config.mdl_config).to(self._config.device)
        self.model.train(mode=True)
        self.model.init_params()

    def _create_optimizer(self) -> Optimizer:
        return AdamW(self.model.parameters(),
                    betas=(0.9, 0.98),
                    weight_decay=10e-9,
                    lr = self._config.learing_rate)

    def _create_scheduler(self, optimizer : Optimizer) -> LRScheduler:
        """Creates the custom learning rate scheduled optimzer.

        Args:
            optimizer (Optimizer): The optimizer that will have its learning rate changed

        Returns:
            LRScheduler
        """

        def _schedule(step : int) -> float:

            # to account for step = 0
            step += 1

            dm = self._config.mdl_config.model_dimension
            warmup_steps = self._config.warmup_steps

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

    def save_checkpoint(self,
                        epoch : int,
                        step : int,
                        optimizer : Optimizer,
                        scheduler : LRScheduler):
        import os
        folder = f'{self.checkpoint_folder}{epoch}/{step}/'
        os.makedirs(folder, exist_ok=True)
        save({
                'epoch' : epoch,
                'model_state_dict' : self.model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'config' : self._config,
            },
            f=open(f'{folder}checkpoint.pt', 'wb+')
        )

    def _log_metrics(self, predictions : Tensor, labels : Tensor, step : int):

        wandb.log({'train_acc' : calculate_accuracy(predictions, labels)}, step=step)
        wandb.log({'blue_score' : decode_and_calculate_bleu_score(predictions, labels, self.model.tokenizer_de)}, step=step)

    def _calculate_num_tkns(self, train_batch : Tensor):
        """Calculate the number of tokens in a batch

        Args:
            train_tokens (Tensor): _description_
        """
        padding_idx = self.model.tokenizer_en.pad_token_id
        train_batch[train_batch == padding_idx] = 0

        return train_batch.count_nonzero()

    def _val_epoch(self, val_dataloader : DataLoader, loss_fn : CrossEntropyLoss):
        """Gets the loss over the val dataset.

        Args:
            val_dataloader (DataLoader):
            loss_fn (CrossEntropyLoss):
        """

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

        # Average the val loss and log
        wandb.log({'val_loss' : sum(val_loss)/len(val_loss)})

    def train(self):

        _ = wandb.init(
            project='transformer-testing',
            config = self._config.dict(),
        )

        self._instantiate_model()
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        loss_fn   = self._get_loss_fn()

        tokens_trained = 0
        val_dataloader = self._create_dataloader('validation', shuffle=False)

        for epoch_num in range(0, self._config.num_epochs):

            # re-shulffles the data by remaking each epoch
            train_dataloader = self._create_dataloader('train', shuffle=True)

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

                optimizer.zero_grad()
                labels = batch['de']['input_ids']

                predictions = self.model(
                    input_tkns=batch['en']['input_ids'],
                    target_tkns=labels)

                loss = loss_fn(
                    # flatten out the predictions and labels
                    predictions,
                    batch['de']['input_ids'].contiguous().view(-1)
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                tokens_trained += self._calculate_num_tkns(batch['en']['input_ids'])

                if i % self._config.logging_freq == 0:
                    wandb.log({'train_loss' : loss}, step = i)
                    wandb.log({'total_tokens_trained' : tokens_trained}, step = i)
                    self._log_metrics(predictions, labels, step= i)

                if i % self._config.val_epoch_freq == 0:
                    self._val_epoch(val_dataloader, loss_fn=loss_fn)

                if i % self._config.checkpoint_steps == 0:
                    self.save_checkpoint(
                        epoch=epoch_num,
                        step=i,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

            print(f'Epoch {epoch_num} complete')