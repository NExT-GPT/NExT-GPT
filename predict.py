import torch

from nextgpt.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from nextgpt.conversation import conv_templates, SeparatorStyle
from nextgpt.model.builder import load_pretrained_model
from nextgpt.utils import disable_torch_init
from nextgpt.mm_utils import tokenizer_image_token, tokenizer_multiple_token
from transformers.generation.streamers import TextIteratorStreamer
import transformers
from dataclasses import dataclass, field
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List

import requests
from io import BytesIO
import scipy
from cog import BasePredictor, Input, Path, ConcatenateIterator
import time
import subprocess
from threading import Thread
from diffusers.utils import export_to_video
import os
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

# url for the weights mirror
REPLICATE_WEIGHTS_URL = None


@dataclass
class GenerateArguments:
    # Basic generation arguments
    top_k: int = field(default=1, metadata={"help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"})
    top_p: float = field(default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."})
    temperature: float = field(default=1.0, metadata={"help": "The value used to module the next token probabilities. Must be strictly positive."},)
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum number of new tokens to generate. The generation process will stop when reaching this threshold."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample from the output distribution to generate new tokens. If False, use argmax."})
    use_cache: bool = field(default=False, metadata={"help": "Whether to cache the hidden states of the model to speed up generation."})
    output_hidden_states: bool = field(default=True,metadata={"help": "Whether to return the hidden states of all intermediate layers."})
    
    # Image inference arguments
    guidance_scale_for_img: float = field(default=7.5, metadata={"help": "The scale for the guidance loss of image signal."})
    num_inference_steps_for_img: int = field(default=50, metadata={"help": "The number of inference steps for image signal."})

    # Video inference arguments
    guidance_scale_for_vid: float = field(default=7.5, metadata={"help": "The scale for the guidance loss of video signal."})
    num_inference_steps_for_vid: int = field(default=50, metadata={"help": "The number of inference steps for video signal."})
    height: int = field(default=320, metadata={"help": "The height of the video frame."})
    width: int = field(default=576, metadata={"help": "The width of the video frame."})
    num_frames: int = field(default=16, metadata={"help": "The number of frames in the video."})

    # Audio inference arguments
    guidance_scale_for_aud: float = field(default=7.5, metadata={"help": "The scale for the guidance loss of audio signal."})
    num_inference_steps_for_aud: int = field(default=50, metadata={"help": "The number of inference steps for audio signal."})
    audio_length_in_s: float = field(default=5.0, metadata={"help": "The length of the audio signal in seconds."})



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                 subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self, model_base, model_name, model_path, load_8bit=False, load_4bit=False) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        disable_torch_init()

        # ./pretrain_ckpt/vicuna-7b-v1.5
        self.tokenizer, self.model, self.image_processor, self.video_processor, self.audio_processor, self.context_len, self.model_config = load_pretrained_model(model_base, model_name, model_path, load_8bit=load_8bit, load_4bit=load_4bit) 
                                    
    def predict(
        self,
        image: str = None,
        prompt: str = None,
        top_p: float = 1.0,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        """Run a single prediction on the model"""

        # prepare generation arguments
        parser = transformers.HfArgumentParser(GenerateArguments)
        generation_args = parser.parse_args_into_dataclasses()[0]

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[[835]], encounters=1)])

        generation_args.top_p = top_p if top_p is not None else generation_args.top_p
        generation_args.temperature = temperature if temperature is not None else generation_args.temperature
        generation_args.max_new_tokens = max_new_tokens if max_new_tokens is not None else generation_args.max_new_tokens
        generation_args.stopping_criteria = stopping_criteria

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
    
        # loop start
    
        # just one turn, always prepend image token
        # inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt  # prepend image token when need to understanding images content
        inp = prompt  # no need to prepend image token when generting images
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        _prompt = conv.get_prompt()
        print("prompt: ", _prompt)
        input_ids = tokenizer_multiple_token(_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        print("input_ids: ", input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        image_signal_token_indices = [self.tokenizer(f"<image_{i:02d}>").input_ids for i in range(self.model_config.n_img_tokens)]
        video_signal_token_indices = [self.tokenizer(f"<video_{i:02d}>").input_ids for i in range(self.model_config.n_vid_tokens)]
        audio_signal_token_indices = [self.tokenizer(f"<audio_{i:02d}>").input_ids for i in range(self.model_config.n_aud_tokens)]
        print("image_signal_token_indices: ", image_signal_token_indices)
        print("video_signal_token_indices: ", video_signal_token_indices)
        print("audio_signal_token_indices: ", audio_signal_token_indices)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                # images=image_tensor,
                image_signal_token_indices=image_signal_token_indices,
                video_signal_token_indices=video_signal_token_indices,
                audio_signal_token_indices=audio_signal_token_indices,
                **generation_args.__dict__
            )
            print("output: ", output)
            print("output shape: ", self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=False)[0])
            for k in output.keys():
                print(k)
                if 'images' == k:
                    for m in output['images']:
                        if isinstance(m, torch.Tensor):
                            print(m)
                        else:
                            if not os.path.exists('./assets/images'):
                                os.mkdir('./assets/images')
                            m[0].save(f'./assets/images/{prompt}.jpg')

                elif 'videos' == k:
                    for idx, m in enumerate(output['videos']):
                        if isinstance(m, torch.Tensor):
                            print(m)
                        else:
                            if not os.path.exists('./assets/videos'):
                                os.mkdir('./assets/videos')
                            video_path = export_to_video(video_frames=m, output_video_path=f'./assets/videos/{prompt}.mp4')
                            print("video path: ", video_path)
                elif 'audios' == k:
                    for idx, m in enumerate(output['audios']):
                        if isinstance(m, torch.Tensor):
                            print(m)
                        else:
                            audio_path = f'./assets/audios/{prompt}.wav'
                            if not os.path.exists('./assets/audios'):
                                os.mkdir('./assets/audios')
                            scipy.io.wavfile.write(audio_path, rate=16000, data=m)
                            print("audio path: ", audio_path)
                else:
                    pass
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup(model_base=None, model_name="nextgpt-v1.5-7b", model_path="./checkpoints/nextgpt-v1.5-7b", load_8bit=False, load_4bit=False)
    # show me a beautiful landscape of 
    # descibe the bird in the image
    predictor.predict(image="./assets/bird_image.jpg", prompt="show me an image of a cute dog running on the grass")

