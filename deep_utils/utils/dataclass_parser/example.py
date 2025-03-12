
from dataclass_argparser import DataClassArgParser, Arg
from dataclasses import dataclass
from enum import Enum

class Action(Enum):
    TRAIN = 'train'
    TEST = 'test'

class Datasets(Enum):
    MNIST = 'mnist'
    CIFAR = 'cifar'
    
class Nets(Enum):
    CNN = 'cnn'
    RESNET = 'resnet'


@dataclass(frozen=True)
class Args:
    """My new cool model"""
    action: DataClassArgParser.Choice[Action]
    n_batches: DataClassArgParser.Int(help="Nubmer of batches") = 100
    save_epoch: DataClassArgParser.Int = 20
    # device: Arg(type=torch.device) = torch.device('cpu')
    dataset: DataClassArgParser.Choice[Datasets] = Datasets.MNIST
    model: DataClassArgParser.Choice[Nets] = Nets.CNN
    num_workers: DataClassArgParser.Int = 8
    seed: DataClassArgParser.Int = 123
    lr: DataClassArgParser.Float = 0.001
    random_state: int = 1234

@dataclass
class CrawlerConfig:
    """Configuration"""
    search_query: str
    max_videos: int = 5
    output_format: str = "json"
    log_level: str = "INFO"

if __name__ == '__main__':
    config = DataClassArgParser.parse_to(CrawlerConfig)
    print(config)
    DataClassArgParser.parse_to(Args, args=['--help'])
# prints
"""
usage: ipython [-h] [--action {Action.TRAIN,Action.TEST}]
               [--n-batches N_BATCHES] [--save-epoch SAVE_EPOCH]
               [--device DEVICE] [--dataset {Datasets.MNIST,Datasets.CIFAR}]
               [--model {Nets.CNN,Nets.RESNET}] [--num-workers NUM_WORKERS]
               [--seed SEED] [--lr LR]

My new cool model

optional arguments:
  -h, --help            show this help message and exit
  --action {Action.TRAIN,Action.TEST}
  --n_batches N_BATCHES
                        Nubmer of batches
  --save-epoch SAVE_EPOCH
  --device DEVICE
  --dataset {Datasets.MNIST,Datasets.CIFAR}
  --model {Nets.CNN,Nets.RESNET}
  --num_workers NUM_WORKERS
  --seed SEED
  --lr LR
  --random_state RANDOM_STATE
"""