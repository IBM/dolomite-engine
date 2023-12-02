# https://nvidia.github.io/apex/optimizers.html
from apex.optimizers import FusedAdam as ApexFusedAdam
from apex.optimizers import FusedLAMB as ApexFusedLAMB
from apex.optimizers import FusedNovoGrad as ApexFusedNovoGrad
from apex.optimizers import FusedSGD as ApexFusedSGD

# https://deepspeed.readthedocs.io/en/latest/optimizers.html
from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam as DeepSpeedFusedAdam
from deepspeed.ops.lamb import FusedLamb as DeepSpeedFusedLAMB
from deepspeed.runtime.fp16.onebit import OnebitAdam as DeepSpeedOnebitAdam
from deepspeed.runtime.fp16.onebit import OnebitLamb as DeepSpeedOnebitLAMB
from deepspeed.runtime.fp16.onebit import ZeroOneAdam as DeepSpeedZeroOneAdam

# https://pytorch.org/docs/stable/optim.html
from torch.optim.adadelta import Adadelta as TorchAdadelta
from torch.optim.adagrad import Adagrad as TorchAdagrad
from torch.optim.adam import Adam as TorchAdam
from torch.optim.adamax import Adamax as TorchAdamax
from torch.optim.adamw import AdamW as TorchAdamW
from torch.optim.asgd import ASGD as TorchASGD
from torch.optim.lbfgs import LBFGS as TorchLBFGS
from torch.optim.nadam import NAdam as TorchNAdam
from torch.optim.radam import RAdam as TorchRAdam
from torch.optim.rmsprop import RMSprop as TorchRMSprop
from torch.optim.rprop import Rprop as TorchRprop
from torch.optim.sgd import SGD as TorchSGD

from engine.optimization.optimizer import get_optimizer
from engine.optimization.scheduler import get_scheduler_method
