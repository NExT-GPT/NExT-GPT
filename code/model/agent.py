import os.path

import torch
from header import *


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])

        self.load_parameters(self.args['save_path'], self.args['stage'])
        
        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
            self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self):
        self.ds_engine.module.eval()
        output = self.ds_engine.generate(self.args)
        return output

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()

        loss, mle_acc, mse_loss = self.ds_engine(batch)

        self.writer.add_scalar('loss', loss, current_step)
        self.writer.add_scalar('mle_acc', mle_acc, current_step)
        # if isinstance(mse_loss, list):
        #     self.writer.add_scalar('img_mse_loss', mse_loss[0], current_step)
        #     self.writer.add_scalar('vid_mse_loss', mse_loss[1], current_step)
        #     self.writer.add_scalar('aud_mse_loss', mse_loss[2], current_step)
        if isinstance(mse_loss, torch.Tensor):
            self.writer.add_scalar('mse_loss', mse_loss, current_step)
        else:
            pass
        # self.writer.add_scalar('mse_loss', mse_loss, current_step)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}; mse_loss: {round(mse_loss[0].item(), 4)} ')
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
            # ; mse_loss: {round(mse_loss[0].item(), 4)}
        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        """
            this function also save the trainable parameters and specific name parameters
        """
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.ds_engine.module.named_parameters()
        }
        state_dict = self.ds_engine.module.state_dict()
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs' in k:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs_video' in k:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs_audio' in k:
                checkpoint[k] = v
            if 'llama_proj' in k:
                checkpoint[k] = v
        torch.save(checkpoint, f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(path)
        # save configuration
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def print_model_parameters(self, use_4bit=False):
        """
            Prints the number of trainable parameters in the model.
            """
        trainable_params = 0
        all_param = 0
        lora = 0
        image = 0
        video = 0
        audio = 0
        linear = 0
        llama = 0
        imagebind = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'gen_text_hidden_fcs_video' in name:
                video += num_params
            elif 'gen_text_hidden_fcs_audio' in name:
                audio += num_params
            elif 'gen_text_hidden_fcs' in name:
                image += num_params
            elif 'llama_proj' in name:
                linear += num_params
            elif 'llama_model' in name:
                llama += num_params
            elif 'visual_encoder' in name:
                imagebind += num_params
            else:
                pass

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(f'lora params: {lora:,d} || video params: {video:,d} || audio params: {audio:,d} || image params: {image:,d}')
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d}')

    def load_parameters(self, path, stage=3):
        if os.path.exists(os.path.join(path, 'pytorch_model.pt')):
            print('loading parameters from {}'.format(self.args['save_path']))
            delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=torch.device('cuda'))
            checkpoint = OrderedDict()
            if stage == 3:
                for k, v in delta_ckpt.items():
                    if 'llama_model.model.embed_tokens.weight' in k:
                        checkpoint['llama_model.base_model.model.model.embed_tokens.weight'] = v
                    elif 'llama_model.lm_head.weight' in k:
                        checkpoint['llama_model.base_model.model.lm_head.weight'] = v
                    else:
                        checkpoint[k] = v
            else:
                checkpoint = delta_ckpt
            self.model.load_state_dict(checkpoint, strict=False)

