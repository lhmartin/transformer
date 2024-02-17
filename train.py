import wandb
from data.wmt14_dataset import WMT14_Dataset
from transformer import Transformer

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from pydantic import BaseModel
from tqdm import tqdm

class Trainer():
    
    class Config(BaseModel):
        learing_rate : float = 0.001
        batch_size : int = 32
        num_epochs : int = 100
        device     : str = 'cuda'
        mdl_config : Transformer.Config
        
    def __init__(self, config : Config) -> None:
        self._config = config
        
    def _instantiate_model(self):
        self.model = Transformer(self._config.mdl_config).to(self._config.device)
        
    def _create_optimizer(self):
        return Adam(self.model.parameters(),
                    lr = self._config.learing_rate,)
        
    def _create_dataset(self, split : str):
        return WMT14_Dataset(split=split)
    
    def _create_dataloader(self, split, shuffle=True) -> DataLoader:

        data = self._create_dataset(split)
        return DataLoader(dataset=data,
                          batch_size=self._config.batch_size,
                          collate_fn=self.model.collate_fn,
                          )
        
    def _get_loss_fn(self) -> CrossEntropyLoss:
        return CrossEntropyLoss(label_smoothing=0.1,ignore_index=self.model.tokenizer_en.pad_token_id)

    def train(self):
        
        run = wandb.init(
            project='transformer-testing',
            config = self._config.dict(),
        )

        self._instantiate_model()
        optimizer = self._create_optimizer()
        loss_fn   = self._get_loss_fn()

        train_dataloader = self._create_dataloader('train', shuffle=True)
        test_dataloader = self._create_dataloader('test', shuffle=False)
        
        for epoch_num in range(0, self._config.num_epochs):
            for batch in tqdm(train_dataloader):

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
                
                wandb.log({'loss' : loss})

            print(f'Epoch {epoch_num} complete')
            

if __name__ == '__main__':
    
    cfg = Trainer.Config(
        mdl_config=Transformer.Config(),
        batch_size=16
    )

    trainer = Trainer(cfg)

    trainer.train()