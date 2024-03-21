import time
import torch
from torchtext.data.metrics import bleu_score
import wandb
from data.wmt14_dataset import WMT14_Dataset, DATASET_SPLITS
from transformer import Transformer

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader
from torch import Tensor, no_grad, save, where, stack, cat, zeros
from pydantic import BaseModel
from tqdm import tqdm
from math import pow
import datetime
from typing import Dict, List, Literal, Tuple

from utils.metrics import decode_and_calculate_bleu_score, calculate_accuracy

class Trainer():
    """
    Manages the training of the Transformer model
    """

    class Config(BaseModel):
        checkpoint_folder : str = './transformer_checkpoints/'
        checkpoint_steps  : int = 30000     # Number of steps to then save a checkpoint
        val_epoch_freq    : int = 50000     # How many steps have to be taken until running through the val dataloader
        learing_rate      : float = 2.0
        translation_dir   : Literal['en_to_de', 'de_to_en'] = 'de_to_en'
        batch_size        : int = 256
        num_epochs        : int = 10
        device            : str = 'cuda'
        logging_freq      : int = 10
        warmup_steps      : int = 4000
        weight_init       : Literal['default', 'xavier'] = 'xavier'
        resume_from_ckpt  : str | None = None
        mdl_config        : Transformer.Config

    def __init__(self, config : Config) -> None:
        self._config = config
        self.checkpoint_folder = self._config.checkpoint_folder + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

        self.trgt_lang =  'de' if self._config.translation_dir == 'en_to_de' else 'en'
        self.src_lang =  'en' if self._config.translation_dir == 'en_to_de' else 'de'

    def _instantiate_model(self):
        self.model = Transformer(self._config.mdl_config).to(self._config.device)
        self.model.train(mode=True)
        self.model.init_params(self._config.weight_init)
        self.source_tokenizer = self.model.tokenizer_en if self._config.translation_dir == 'en_to_de' else self.model.tokenizer_de
        self.trgt_tokenizer = self.model.tokenizer_de if self._config.translation_dir == 'en_to_de' else self.model.tokenizer_en

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

    def _get_loss_fn(self) -> KLDivLoss:
        return KLDivLoss(reduction='batchmean')

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

    def _log_metrics(self, predictions : Tensor, labels : Tensor, stage : str):

        wandb.log({f'{stage}_acc' : calculate_accuracy(predictions, labels)})
        wandb.log({f'{stage}_bleu_score' : decode_and_calculate_bleu_score(predictions, labels, self.trgt_tokenizer)})

    def _calculate_num_tkns(self, train_batch : Tensor):
        """Calculates the number of tokens in a batch.

        Args:
            train_tokens (Tensor): Value
        """
        padding_idx = self.model.tokenizer_en.pad_token_id
        train_batch[train_batch == padding_idx] = 0

        return train_batch.count_nonzero()

    def greedy_decode_bleu_score(self, batch):

        src_token_ids, trg_token_ids = batch[self.src_lang]['input_ids'], batch[self.trgt_lang]['input_ids']

        src_mask = self.model.make_source_mask(src_token_ids)
        src_representations_batch = self.model.encode(src_token_ids, src_mask)

        predicted_sentences = self.model.greedy_decoding(src_representations_batch,
                                                         src_mask,
                                                         self.trgt_tokenizer,
                                                         max_target_tokens=self._config.mdl_config.max_sequence_len)

        predicted_sentences_corpus = [[sent] for sent in predicted_sentences]  # add them to the corpus of translations

        # Get the token and not id version of GT (ground-truth) sentences
        gt_sentences_corpus = self.trgt_tokenizer.batch_decode(trg_token_ids.cpu(),
                                                                    skip_special_tokens=True)  # add them to the corpus of GT translations

        return bleu_score(predicted_sentences_corpus, gt_sentences_corpus)

    def _val_epoch(self, loss_fn : KLDivLoss):
        """
        Calculate metrics and loss over the validation dataset.
        """
        # Set the model into evaluation mode
        self.model.eval()
        val_dataloader = self._create_dataloader('validation', shuffle=False)

        with torch.no_grad():

            total_val_loss = []
            total_greedy_bleu_score = []

            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

                if i % 10 == 0:
                    total_greedy_bleu_score.append(self.greedy_decode_bleu_score(batch))

                input_tkns, model_trg_in, model_trg_gt = self._add_labels_and_inputs(batch)
                OH_labels = self.one_hot_labels(model_trg_gt.contiguous().reshape(-1, 1))

                predictions = self.model(
                    input_tkns=input_tkns,
                    target_tkns=model_trg_in)

                loss = loss_fn(
                    # flatten out the predictions and labels
                    predictions,
                    OH_labels
                )

                total_val_loss.append(loss.cpu())
                self._log_metrics(predictions, model_trg_gt, 'train')

        def _avg(vals : List[float]):
            return sum(vals)/len(vals)

        wandb.log({'total_val_loss' : _avg(total_val_loss)})
        wandb.log({'val_bleu_score' : _avg(total_greedy_bleu_score)})
        self.model.train()

    def _shift_labels(self, label_batch : Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:

        model_target_in = []
        model_target_gt = []

        for row in label_batch['input_ids']:
            if 0 in row:
                pad_idx = where(row == 0)[0][0]
            else:
                pad_idx = len(row)

            non_padded = row[:pad_idx]
            model_target_in.append(cat((non_padded[:-1], row[pad_idx:])))
            model_target_gt.append(cat((non_padded[1:], row[pad_idx:])))

        return stack(model_target_in), stack(model_target_gt)

    def _add_labels_and_inputs(self, batch : Dict[str, Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:

        if self._config.translation_dir == 'de_to_en':
            model_target_in, model_target_gt = self._shift_labels(batch['en'])
            src_tokens = batch['de']['input_ids']
        elif self._config.translation_dir == 'en_to_de':
            model_target_in, model_target_gt = self._shift_labels(batch['de'])
            src_tokens = batch['en']['input_ids']
        else:
            raise ValueError('Unknown translation dir')

        return src_tokens, model_target_in, model_target_gt

    def one_hot_labels(self, labels : Tensor):

        batch_size = labels.shape[0]

        OH_tokens = torch.zeros((batch_size, self._config.mdl_config.tgt_vocab_size), device=self.model.device)
        OH_tokens.scatter_(1, labels, 1.0)
        OH_tokens[:, 0] = 0

        return OH_tokens

    def resume(self,
               ckpt : Dict[str, Tensor | int],
               optimizer : Optimizer,
               scheduler : LRScheduler):

        self.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        return optimizer, scheduler

    def train(self, ckpt : Dict[str, Tensor | int] | None = None):

        _ = wandb.init(
            project='transformer-testing',
            config = self._config.dict(),
        )

        self._instantiate_model()
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        loss_fn   = self._get_loss_fn()

        if ckpt:
            optimizer, scheduler = self.resume(ckpt,
                                               scheduler=scheduler,
                                               optimizer=optimizer)

        tokens_trained = 0

        for epoch_num in range(0, self._config.num_epochs):

            # re-shulffles the data by remaking each epoch
            train_dataloader = self._create_dataloader('train', shuffle=True)

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

                optimizer.zero_grad()
                input_tkns, model_trg_in, model_trg_gt = self._add_labels_and_inputs(batch)

                OH_labels = self.one_hot_labels(model_trg_gt.contiguous().reshape(-1, 1))

                predictions = self.model(
                    input_tkns=input_tkns,
                    target_tkns=model_trg_in)

                loss = loss_fn(
                    # flatten out the predictions and labels
                    predictions,
                    OH_labels
                )

                loss.backward()
                optimizer.step()
                scheduler.step()

                tokens_trained += self._calculate_num_tkns(input_tkns)

                if i % self._config.logging_freq == 0:
                    wandb.log({'train_loss' : loss},)
                    wandb.log({'total_tokens_trained' : tokens_trained})
                    self._log_metrics(predictions, model_trg_gt, 'train')

                if i % self._config.val_epoch_freq == 0 and i != 0:
                    self._val_epoch(loss_fn)

                if i % self._config.checkpoint_steps == 0:
                    self.save_checkpoint(
                        epoch=epoch_num,
                        step=i,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

            print(f'Epoch {epoch_num} complete')