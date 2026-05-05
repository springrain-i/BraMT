import os
import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import datetime

from utils.util import generate_mask
from utils.logger import ModelLogger

# Optional: Weights & Biases integration
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        # create experiment name containing model info and timestamp
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        experiment_name = f"{timestamp}_pretrain_{model.__class__.__name__}"
        # initialize logger
        self.logger = ModelLogger(params=self.params, experiment_name=experiment_name, monitor_key='loss', monitor_mode='min')
        # log config and model architecture

        config_dict = vars(params) if hasattr(params, '__dict__') else params
        self.logger.log_experiment_config(config_dict)
        self.logger.log_model_architecture(model)

        # initialize wandb if enabled and available
        self.wandb_run = None
        self.wandb_enabled = bool(getattr(self.params, 'use_wandb', False)) and _WANDB_AVAILABLE and getattr(self.params, 'wandb_mode', 'online') != 'disabled'
        if self.wandb_enabled:
            try:
                # optional programmatic login if key provided
                api_key = getattr(self.params, 'wandb_api_key', None)
                if api_key:
                    os.environ["WANDB_API_KEY"] = str(api_key)
                    try:
                        wandb.login(key=str(api_key), relogin=True)
                    except Exception:
                        pass
                wandb_kwargs = dict(
                    project=getattr(self.params, 'wandb_project', 'eeg-pretrain'),
                    name=experiment_name,
                    mode=getattr(self.params, 'wandb_mode', 'online'),
                    dir=getattr(self.params, 'wandb_dir', './wandb'),
                    config=config_dict,
                )
                entity = getattr(self.params, 'wandb_entity', None)
                if entity:
                    wandb_kwargs['entity'] = entity
                self.wandb_run = wandb.init(**wandb_kwargs)
                wandb.watch(self.model, log='gradients', log_freq=100)
            except Exception:
                self.wandb_run = None
                self.wandb_enabled = False
  

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        summary(self.model, input_size=(1, 19, 30, 200))

        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler=='ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler=='StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5*self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler=='MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10*self.data_length, 20*self.data_length, 30*self.data_length], gamma=0.1
            )
        elif self.params.lr_scheduler=='CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001, step_size_up=self.data_length*5,
                step_size_down=self.data_length*2, mode='exp_range', gamma=0.9, cycle_momentum=False
            )


    def train(self):
        best_loss = 10000
        for epoch in range(self.params.epochs):
            start_time = timer()
            losses = []
            for step, x in enumerate(tqdm(self.data_loader, mininterval=10), start=1):
                self.optimizer.zero_grad()
                x = x.to(self.device) / 100
                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    y = self.model(x, mask=mask)
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    loss = self.criterion(masked_y, masked_x)

                    # non_masked_x = x[mask == 0]
                    # non_masked_y = y[mask == 0]
                    # non_masked_loss = self.criterion(non_masked_y, non_masked_x)
                    # loss = 0.8 * masked_loss + 0.2 * non_masked_loss
                else:
                    y = self.model(x)
                    loss = self.criterion(y, x)

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())

                # step-level wandb logging
                if self.wandb_enabled:
                    try:
                        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                        global_step = (self.data_length * epoch) + step
                        wandb.log({'loss': float(loss.item()), 'lr': float(current_lr), 'epoch': epoch + 1, 'step': global_step})
                    except Exception:
                        pass
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            elapsed_mins = (timer() - start_time) / 60
            # console print
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}, Time: {elapsed_mins:.2f} mins')
            # logger record
            try:
                self.logger.log_training_step(epoch=epoch+1,
                                              step=self.data_length * (epoch + 1),
                                              loss=mean_loss,
                                              lr=learning_rate,
                                              metrics={'training_time_mins': elapsed_mins})
                # also call validation-style log to update monitored best
                self.logger.log_validation_results(epoch + 1, mean_loss, {'loss': mean_loss})
            except Exception:
                pass

            # epoch-level wandb logging
            if self.wandb_enabled:
                try:
                    wandb.log({'epoch_mean_loss': float(mean_loss), 'lr': float(learning_rate), 'epoch': epoch + 1})
                except Exception:
                    pass

            # save best model
            if mean_loss < best_loss:
                if not os.path.isdir(self.params.model_dir):
                    os.makedirs(self.params.model_dir, exist_ok=True)
                model_path = os.path.join(self.params.model_dir, f'epoch{epoch+1}_loss{mean_loss:.6f}.pth')
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
                # log best save

                self.logger.train_logger.info(f"Saved best pretrain model to {model_path} at epoch {epoch+1} (loss={mean_loss:.6f})")

                best_loss = mean_loss

                # wandb best metric
                if self.wandb_enabled:
                    try:
                        wandb.log({'best_loss': float(best_loss), 'best_epoch': epoch + 1})
                    except Exception:
                        pass

        # finish wandb run at the end of training
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass
