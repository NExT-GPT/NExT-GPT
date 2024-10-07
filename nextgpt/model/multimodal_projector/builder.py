# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import torch
import torch.nn as nn
from .projector import *


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_input_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_input_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(in_features=config.mm_hidden_size, out_features=config.hidden_size)
    
    if projector_type == "mlp":
        return MLP(in_features=config.mm_hidden_size, out_features=config.hidden_size, num_layers=2)
    
    if projector_type == "group":
        return GroupProjector(in_features=config.mm_hidden_size, out_features=config.hidden_size, **kwargs)

    if projector_type == "identity":
        return IdentityMap()
    
    raise ValueError(f"Unknown INPUT projector type: {projector_type}")


def build_output_projector(config, projector_type, out_features, num_query_token, delay_load=False, **kwargs):
    # projector_type = getattr(config, "mm_output_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(in_features=config.hidden_size, out_features=out_features)
    
    if projector_type == "qformer":
        return QFormer(in_features=config.hidden_size, out_features=out_features, num_query_token=num_query_token)
    
    if projector_type == "transformer":
        return TransformersProjector(in_features=config.hidden_size, out_features=out_features, num_query_token=num_query_token)
    if projector_type == "identity":
        return IdentityMap()
    raise ValueError(f"Unknown OUTPUT projector type: {projector_type}")


if __name__ == "__main__":

    class Config:
        mm_hidden_size = 1024
        hidden_size = 768
        mm_input_projector_type = "group"
    
    config = Config()
    input_projector = build_input_projector(config)
    print(input_projector)
    x = torch.randn(2, 128, 1024)
    input_emb = torch.randn(2, 128, 1024)
    print(input_projector(x, input_emb).shape)

    # output_projector = build_output_projector(config, projector_type="transformer", out_features=1024, num_query_token=10)
