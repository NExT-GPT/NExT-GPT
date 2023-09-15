import torch
from torch import nn
from .qformer import BertLMHeadModel, BertConfig
# from header import *

class TextFcLayer(nn.Module):
    """Layers used in mapping text embeddings to visual outputs."""

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2, cross_attention_freq=1):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = num_hidden_layers
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1,
                 mode: str = 'linear',
                 freeze_qformer=False):
        """
        :param mode: ['linear', 'transformer', 'qformer']
        :param freeze_qformer: whether freeze the weights of qformer
        """
        super().__init__()

        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.mode = mode
        self.out_dim = out_dim

        if mode == 'linear':
            self.model = nn.Linear(in_dim, out_dim)
        elif mode == 'transformer':
            hidden_dim = 512
            self.fc = nn.Linear(in_dim, hidden_dim)
            self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                      d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                      dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
            self.model = nn.Linear(hidden_dim, out_dim)
            self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
        elif mode == 'qformer':
            # raise NotImplementedError(mode)  # TODO: ADD Q-former FOR MAPPING LAYER
            print('Loading Q-Former')
            hidden_dim = 768
            self.fc = nn.Linear(in_dim, hidden_dim)
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_output_tokens, hidden_dim
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            # self.load_from_pretrained(url_or_filename=q_former_model)
            self.model = nn.Linear(hidden_dim, out_dim)
            # if freeze_qformer:
            #     for name, param in self.Qformer.named_parameters():
            #         param.requires_grad = False
            #     self.Qformer = self.Qformer.eval()
            #     # self.Qformer.train = disabled_train
            #     self.query_tokens.requires_grad = False
            #     # logging.info("freeze Qformer")
            print('Loading Q-Former Done')

        else:
            raise NotImplementedError(mode)

    def forward(self, x: torch.Tensor, input_embs: torch.Tensor) -> torch.Tensor:
        outputs = None

        if isinstance(self.model, nn.ModuleList):
            assert len(self.model) == x.shape[1] == self.num_input_tokens, (
            len(self.model), x.shape, self.num_input_tokens)
            outputs = []
            for i in range(self.num_input_tokens):
                outputs.append(self.model[i](x[:, i, :]))  # (N, D)
            outputs = torch.stack(outputs, dim=1)  # (N, T_I_V_A.txt, D)
        elif self.mode == 'transformer':
            # print("x.size: ", x.size())
            # print("input_embs.size: ", input_embs.size())
            x = x + input_embs
            # print('layer x: ', x)
            x = self.fc(x)
            # print('layer fc x: ', x)
            x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
            # print('layer tfm x: ', x)
            outputs = self.model(x)
            # print('layer tfm model: ', x)

            if outputs.shape[1] != self.num_output_tokens and self.mode == 'linear':
                if self.mode == 'linear':
                    outputs = outputs[:, :self.num_output_tokens, :]
                else:
                    raise NotImplementedError
        elif self.mode == 'qformer':
            x = x + input_embs
            x = self.fc(x)
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
            outputs = self.model(outputs)

        assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * self.out_dim), (
        outputs.shape, self.num_output_tokens)
        return outputs  # (N, T_I_V_A.txt, D)

