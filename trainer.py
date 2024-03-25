import wandb
import datetime
from transformer import Transformer

from torch import Tensor, no_grad, save, where, stack, cat, zeros
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.nn import KLDivLoss, CrossEntropyLoss
from torch.utils.data import DataLoader

from pydantic import BaseModel
from tqdm import tqdm
from math import pow
from typing import Dict, List, Literal, Tuple
from torchtext.data.metrics import bleu_score

from data import ILSWT17_Dataset, WMT14_Dataset, DATASETS
from utils.metrics import decode_and_calculate_bleu_score, calculate_accuracy, greedy_decode_bleu_score, one_hot_labels

from torch.cuda import set_device
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class Trainer():
    """
    Manages the training of the Transformer model.
    """

    class Config(BaseModel):
        checkpoint_folder : str = './transformer_checkpoints/'
        checkpoint_steps  : int = 30000     # Number of steps to then save a checkpoint
        val_epoch_freq    : int = 50000     # How many steps have to be taken until running through the val dataloader
        learing_rate      : float = 2.0
        loss_fn           : Literal['kl_div', 'cross_entropy'] = 'kl_div'
        dataset           : DATASETS = 'ILSWT17'
        translation_dir   : Literal['en_to_de', 'de_to_en'] = 'de_to_en'
        batch_size        : int = 256
        num_epochs        : int = 10
        device            : str = 'cuda'
        gpus              : List[int] = [0]
        logging_freq      : int = 10
        warmup_steps      : int = 4000
        weight_init       : Literal['default', 'xavier'] = 'xavier'
        resume_from_ckpt  : str | None = None   # Path to the training checkpoint.
        mdl_config        : Transformer.Config

    def __init__(self, config : Config) -> None:
        self._config = config
        self.checkpoint_folder = self._config.checkpoint_folder + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

        self.trgt_lang     = 'de' if self._config.translation_dir == 'en_to_de' else 'en'
        self.src_lang      = 'en' if self._config.translation_dir == 'en_to_de' else 'de'
        self.epoch         = 0
        self.multi_gpu_mode = len(self._config.gpus) > 1
        self.gpu_id = None if self.multi_gpu_mode else self._config.gpus[0]

    def ddp_setup(self, rank : int, world_size : int):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        set_device(rank)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def _instantiate_model(self):
        """
        Initializes the model class and weights. Sets up the appropiate tokenizers.
        Moves the model to appropiate device. If in multi-GPU mode, sets up a DDP instance.
        """
        self.model = Transformer(self._config.mdl_config)

        self.source_tokenizer = self.model.tokenizer_en if self._config.translation_dir == 'en_to_de' else self.model.tokenizer_de
        self.trgt_tokenizer = self.model.tokenizer_de if self._config.translation_dir == 'en_to_de' else self.model.tokenizer_en

        self.model.train(mode=True)
        self.model.init_params(self._config.weight_init)

        if self.multi_gpu_mode:
            self.model.to(f'cuda:{self.gpu_id}')
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            if self._config.device != 'cpu':
                self.model.to(f'cuda:{self.gpu_id}')


    def _create_optimizer(self) -> Optimizer:
        """Instantiates and AdamW optimizer."""
        return AdamW(self.model.parameters(),
                    betas=(0.9, 0.98),
                    weight_decay=10e-9,
                    lr = self._config.learing_rate)

    def _create_scheduler(self, optimizer : Optimizer) -> LRScheduler:
        """Creates the custom learning rate scheduled optimizer.

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

    def _create_dataset(self, split : str) -> ILSWT17_Dataset | WMT14_Dataset:
        """Creates the approtite dataset split for the dataset that is
        defined in the config.

        Args:
            split (str): The split of data that is to be retrieved.
            Can be 'train', 'test' or 'validation'

        Returns:
            _type_: _description_
        """
        if self._config.dataset == 'ILSWT17':
            return ILSWT17_Dataset(split=split)
        elif self._config.dataset == 'WMT14':
            return WMT14_Dataset(split=split)
        else:
            raise ValueError(f'{self._config.dataset} is not a supported dataset.')

    def _create_dataloader(self, split : str, shuffle : bool = True) -> DataLoader:
        """Insantiates and returns a pytorch Dataloader, for the appropiate data split.

        Args:
            split (str): Which portion of the data to use, (eg: train/test/validation)
            shuffle (bool, optional): Whether or not to randomly shuffle the data. Defaults to True.
        """
        data = self._create_dataset(split)

        # if in multi gpu mode, need to instead get the collate_fn from the module
        collate_fn = self.model.collate_fn if not self.multi_gpu_mode else self.model.module.collate_fn

        return DataLoader(dataset=data,
                          batch_size=self._config.batch_size,
                          collate_fn=collate_fn,
                          shuffle= False if self.multi_gpu_mode else shuffle,
                          sampler= DistributedSampler(data) if self.multi_gpu_mode else None
                          )

    def _get_loss_fn(self) -> KLDivLoss | CrossEntropyLoss:
        """Instantiates and returns the loss function instance.

        Raises:
            ValueError: If the configured loss function is not supported

        Returns:
            KLDivLoss | CrossEntropyLoss
        """
        if self._config.loss_fn == 'kl_div':
            return KLDivLoss(reduction='batchmean')
        elif self._config.loss_fn == 'cross_entropy':
            return CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        else:
            raise ValueError(f'{self._config.loss_fn} is not a valid loss function.')

    def calc_loss(self, predictions_batch : Tensor, label_batch : Tensor, loss_fn : KLDivLoss | CrossEntropyLoss) -> Tensor:
        """Calculates the loss between `pred_batch` and `label_batch` using the given `loss_fn`.

        Args:
            predictions_batch (Tensor): The model's preductions shape : [batch_size x seq_len, num_trg_tokens]
            label_batch (Tensor): Ground truth labels shape : [batch_size, seq_len]
            loss_fn (KLDivLoss | CrossEntropyLoss): Function that is used to calculate loss

        Returns:
            Tensor: The loss over the batch
        """
        # reshape from [batch_size, seq_len, 1] to [batch_size x seq_len], to match model output
        label_batch = label_batch.contiguous().reshape(-1)

        if self._config.loss_fn == 'kl_div':
            # kl divergence needs the labels to be the same shape as the predctions, this function transforms
            # the lable indexes into a vector a vector of size [num_trg_tokens, 1] where the value at the index 
            # of the target token is 1 and 0 everywhere else.
            label_batch = one_hot_labels(label_batch.reshape(-1,1), self._config.mdl_config.tgt_vocab_size)

        return loss_fn(predictions_batch, label_batch)

    def save_checkpoint(self,
                        epoch : int,
                        step : int,
                        optimizer : Optimizer,
                        scheduler : LRScheduler):

        folder = f'{self.checkpoint_folder}{epoch}/{step}/'
        os.makedirs(folder, exist_ok=True)
        save({
                'epoch' : epoch,
                'model_state_dict' : self.model.state_dict() if not self.multi_gpu_mode else self.model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'config' : self._config,
            },
            f=open(f'{folder}checkpoint.pt', 'wb+')
        )

    def calc_metrics(self, predictions : Tensor, labels : Tensor) -> Tuple[Tensor, float]:
        return calculate_accuracy(predictions, labels), \
            decode_and_calculate_bleu_score(predictions, labels, self.trgt_tokenizer)

    def _log_metrics(self, predictions : Tensor, labels : Tensor, stage : str) -> None:
        """Logs metrics to weights and biases, currently support two metrics.
        Accuracy - A measurement of how many tokens were correctly predicted.
        blue_score - A measurement of predicted text quality (https://en.wikipedia.org/wiki/BLEU)

        Args:
            predictions_batch (Tensor): The model's preductions shape : [batch_size x seq_len, num_trg_tokens]
            label_batch (Tensor): Ground truth labels shape : [batch_size, seq_len]
            stage (str): What stage of training, will be added to logged metric name.
        """

        acc, bleu_score = self.calc_metrics(predictions, labels)

        wandb.log({f'{stage}_acc' : acc})
        wandb.log({f'{stage}_bleu_score' : bleu_score})

    def _calculate_num_tkns(self, train_batch : Tensor) -> Tensor:
        """Calculates the number of non-padding tokens in a batch.

        Args:
            train_tokens (Tensor): batch of input tokens.
        """
        padding_id = self.source_tokenizer.pad_token_id
        train_batch[train_batch == padding_id] = 0

        return train_batch.count_nonzero()

    def _shift_labels(self, label_batch : Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Moves the labels for input to the model and for the loss function.
        For example, given a target of ['<START>', 'My', 'name' 'is' 'Luke' '<END>'].
        The tokens that will go into the transformer need to only come from before the
        token that should be predicted.

            eg: ['<START>'] -> ['My']
            eg: ['<START>', 'My'] -> ['name']
            ...
            eg: ['<START>', 'My', 'name' 'is' 'Luke'] -> ['<END>']

        This shift is essential, otherwise the model can effectively look at the future
        tokens and will not learn anything.

        Therefore we need to shift the labels to reflect that, to as follows:
            model input:  ['<START>', 'My', 'name' 'is' 'Luke']
            target:       ['My', 'name' 'is' 'Luke', '<END>']

        Args:
            label_batch (Dict[str, Tensor]): A batch of target input ids

        Returns:
            Tuple[Tensor, Tensor]: The model inputs, and targets
        """

        model_target_in = []
        model_target_target = []

        for row in label_batch['input_ids']:
            if 0 in row:
                pad_idx = where(row == 0)[0][0]
            else:
                pad_idx = len(row)

            non_padded = row[:pad_idx]
            # drop the last token, and re-add padding
            model_target_in.append(cat((non_padded[:-1], row[pad_idx:])))

            # drop the first token, and re-add padding
            model_target_target.append(cat((non_padded[1:], row[pad_idx:])))

        return stack(model_target_in), stack(model_target_target)

    def resume(self,
               ckpt : Dict[str, Tensor | int],
               optimizer : Optimizer,
               scheduler : LRScheduler) -> None:
        """Resume training from a checkpoint.

        Args:
            ckpt (Dict[str, Tensor  |  int]): The dictionaty containing all the state
            optimizer (Optimizer): The optimizer that will be updated
            scheduler (LRScheduler): The scheduler that will be update
        """

        self.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        self.epoch = ckpt['epoch']

        return optimizer, scheduler

    def train(self, rank : int, world_size : int, ckpt : Dict[str, Tensor | int] | None = None):
        if self.multi_gpu_mode:
            self.ddp_setup(rank, world_size)
            self.gpu_id = rank

        if rank == 0:
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
            if self.multi_gpu_mode:
                train_dataloader.sampler.set_epoch(epoch_num)

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

                optimizer.zero_grad()

                model_target_in, model_target_gt = self._shift_labels(batch[self.trgt_lang])
                input_tokens = batch[self.src_lang]['input_ids']

                predictions = self.model(
                    input_tkns=input_tokens,
                    target_tkns=model_target_in)

                loss = self.calc_loss(predictions, model_target_gt, loss_fn)

                loss.backward()
                optimizer.step()
                scheduler.step()

                tokens_trained += self._calculate_num_tkns(model_target_in)

                if i % self._config.logging_freq == 0 and rank == 0:
                    wandb.log({'train_loss' : loss},)
                    wandb.log({'total_tokens_trained' : tokens_trained})
                    self._log_metrics(predictions, model_target_gt, 'train')

                if i % self._config.val_epoch_freq == 0 and i != 0 and rank == 0:
                    self._val_epoch(loss_fn)

                if i % self._config.checkpoint_steps == 0 and rank == 0:
                    self.save_checkpoint(
                        epoch=epoch_num,
                        step=i,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

            if rank == 0:
                self._val_epoch(loss_fn)

                self.save_checkpoint(
                            epoch=epoch_num,
                            step=len(train_dataloader),
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )

            print(f'Epoch {epoch_num} complete')

        if self.multi_gpu_mode:
            destroy_process_group()

    def _val_epoch(self, loss_fn : KLDivLoss):
        """
        Calculate metrics and loss over the validation dataset.
        """
        # Set the model into evaluation mode
        self.model.eval()
        val_dataloader = self._create_dataloader('validation', shuffle=False)

        with no_grad():

            total_val_loss = []
            total_greedy_bleu_score = []
            total_acc = []
            total_bleu = []

            for batch in tqdm(val_dataloader, total=len(val_dataloader)):

                total_greedy_bleu_score.append(greedy_decode_bleu_score(
                    batch,
                    self.src_lang,
                    self.trgt_lang,
                    self.model,
                    self.trgt_tokenizer,
                    multi_gpu_mode=self.multi_gpu_mode,
                    ))

                model_target_in, model_target_gt = self._shift_labels(batch[self.trgt_lang])
                src_tokens = batch[self.src_lang]['input_ids']

                predictions = self.model(
                    input_tkns=src_tokens,
                    target_tkns=model_target_in)

                loss = self.calc_loss(predictions, model_target_gt, loss_fn)

                total_val_loss.append(loss.cpu())
                acc, bleu = self.calc_metrics(predictions, model_target_gt)

                total_acc.append(acc)
                total_bleu.append(bleu)

        def _avg(vals : List[float]):
            return sum(vals)/len(vals)

        wandb.log({'total_val_loss' : _avg(total_val_loss)})
        wandb.log({'val_greedy_decode_bleu_score' : _avg(total_greedy_bleu_score)})
        wandb.log({'total_val_acc' : _avg(total_acc)})
        wandb.log({'val_bleu_score' : _avg(total_bleu)})
        self.model.train()
