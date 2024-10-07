import torch
import datetime
import types
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import transformers
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import re
import math
import random
import json
import time
import logging
from omegaconf import OmegaConf
from copy import deepcopy
import ipdb
import argparse
import data
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, TaskType, get_peft_model
from diffusers.utils import export_to_video
import scipy
from torch.utils.tensorboard import SummaryWriter

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
