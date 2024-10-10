
# import paddle
# from paddlemix.models.blip2.Qformer import BertLMHeadModel
# from paddlenlp.transformers.bert.configuration import BertConfig
# from paddle.nn import Transformer

import torch 
import torch.nn as nn
from .qformer import BertLMHeadModel, BertConfig
from .group import GroupingBlock, GroupingLayer, MixerMlp


class MLP(nn.Module):
    def __init__(self, in_features=None, out_features=None, num_layers=1):
        super().__init__()
        modules = [nn.Linear(in_features=in_features, out_features=out_features)]

        for _ in range(1, num_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(in_features=out_features, out_features=out_features))

        self.layer =  nn.Sequential(*modules)
    
    def forward(self, x):
        return self.layer(x)
    
    @property
    def config(self):
        return {"mm_projector_type": "mlp"}
    
    @property
    def device(self):
        return self.layer[0].weight.device
    
    @property
    def dtype(self):
        return self.layer[0].weight.dtype


class GroupProjector(nn.Module):
    def __init__(self, in_features, out_features,
                 num_patches=128,
                 embed_factors=[1, 1, 1], 
                 num_heads=[8, 8, 8], 
                 num_group_tokens=[64, 8, 0],
                 num_output_groups=[64, 8], 
                 norm_layer=nn.LayerNorm, 
                 hard_assignment=True,
                 qkv_bias=False, 
                 depths=[6, 3, 3], 
                 qk_scale=None, 
                 drop_rate=0.0,
                 attn_drop_rate=0.0, 
                 drop_path_rate=0.0, 
                 mlp_ratio=4.0, 
                 use_checkpoint=False):
        super().__init__()

        self.in_fc = nn.Linear(in_features, 512)

        self.embed_dim = 512
        self.embed_factors = embed_factors
        self.num_heads = num_heads
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.norm_layer = nn.LayerNorm
        self.hard_assignment = hard_assignment
        self.qkv_bias = qkv_bias
        self.depths = depths
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        
        self.num_layers = len(depths)
        num_input_token = num_patches
        num_output_token = num_input_token
        num_features = int(self.embed_dim * self.embed_factors[len(depths) - 1])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(self.embed_dim * embed_factors[i_layer])
            downsample = None
            if i_layer < self.num_layers - 1:
                out_dim = self.embed_dim * embed_factors[i_layer + 1]
                downsample = GroupingBlock(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads[i_layer],
                    num_group_token=num_group_tokens[i_layer],
                    num_output_group=num_output_groups[i_layer],
                    norm_layer=norm_layer,
                    hard=hard_assignment,
                    gumbel=hard_assignment)
                num_output_token = num_output_groups[i_layer]

            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(self.embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    norm_layer(prev_dim),
                    MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))

                if dim != prev_dim:
                    group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                    nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None
            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token
        
        self.norm = norm_layer(num_features)
        self.out_fc = nn.Linear(num_features, out_features)

    def forward(self, x, return_attn=False):
        x = self.in_fc(x)
        group_token = None
        attn_dict_list = []
        for layer in self.layers:
            x, group_token, attn_dict = layer(x, group_token, return_attn=return_attn)
            attn_dict_list.append(attn_dict)
        x = self.norm(x)
        x = self.out_fc(x)
        return x
    
    @property
    def config(self):
        return {"mm_projector_type": "group_projector"}
        
    @property
    def device(self):
        return self.in_fc.weight.device
    
    @property
    def dtype(self):
        return self.in_fc.weight.dtype


class QFormer(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, cross_attention_freq=1, num_hidden_layers=2):
        super().__init__()

        self.in_fc = nn.Linear(in_features, out_features)

        qformer_config = BertConfig.from_pretrained('bert-base-uncased')
        qformer_config.encoder_width = out_features
        qformer_config.add_cross_attention = True
        qformer_config.num_hidden_layers = num_hidden_layers
        qformer_config.cross_attention_freq = cross_attention_freq
        qformer_config.gradient_checkpointing = False
        qformer_config.query_length = num_query_token
        qformer_config.use_fusedlinear =False

        self.Qformer = BertLMHeadModel.from_pretrained('bert-base-uncased', config=qformer_config)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

        self.out_fc = nn.Linear(out_features, out_features)

    def forward(self, x, input_embs):
        x = x + input_embs
        x = self.in_fc(x)
        image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(x.device)
        # print(x.size())
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        # print(image_atts.size())
        # print(query_tokens.size())
        outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        # print(outputs.size())
        outputs = self.out_fc(outputs)
        return outputs
    
    @property
    def config(self):
        return {"mm_projector_type": "qformer"}
    
    @property
    def device(self):
        return self.query_tokens.device
    
    @property
    def dtype(self):
        return self.query_tokens.dtype


class TransformersProjector(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, **kwargs):
        super().__init__()
        hidden_dim = 512
        self.in_fc = nn.Linear(in_features, hidden_dim)
        self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                      d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                      dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
        self.out_fc = nn.Linear(hidden_dim, out_features)

        self.query_embs = nn.Parameter(torch.randn(1, num_query_token, hidden_dim))
        self.query_embs.data.normal_(mean=0.0, std=0.0)

    def forward(self, x, input_embs):
        x = x + input_embs
        # print('layer x: ', x)
        x = self.in_fc(x)
        # print('layer fc x: ', x.shape)
        # print('layer fc query_embs: ', self.query_embs.shape)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        # print('layer tfm x: ', x)
        outputs = self.out_fc(x)
        return outputs

    @property
    def config(self):
        return {"mm_projector_type": "transformer"}

    @property
    def device(self):
        return self.query_embs.device
    
    @property
    def dtype(self):
        return self.query_embs.dtype