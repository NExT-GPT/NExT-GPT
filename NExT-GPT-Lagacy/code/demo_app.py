from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
import gradio as gr
import mdtex2html
from model.anyToImageVideoAudio import NextGPTModel
import torch
import json
import tempfile
from PIL import Image
import scipy
from config import *
import imageio
import argparse
import re

# init the model

parser = argparse.ArgumentParser(description='train parameters')
parser.add_argument('--model', type=str, default='nextgpt')
parser.add_argument('--nextgpt_ckpt_path', type=str)  # the delta parameters trained in each stages
parser.add_argument('--stage', type=int, default=3)
args = parser.parse_args()
args = vars(args)
args.update(load_config(args))
model = NextGPTModel(**args)
delta_ckpt = torch.load(os.path.join(args['nextgpt_ckpt_path'], f'pytorch_model.pt'), map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 7b model over ...')

g_cuda = torch.Generator(device='cuda').manual_seed(13)

filter_value = -float('Inf')
min_word_tokens = 10
gen_scale_factor = 4.0
stops_id = [[835]]
ENCOUNTERS = 1
load_sd = True
generator = g_cuda

max_num_imgs = 1
max_num_vids = 1
height = 320
width = 576

max_num_auds = 1
max_length = 246

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text, image_path, video_path, audio_path):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    outputs = text
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines) + "<br>"
    res_text = ''
    split_text = re.split(r' <|> ', text)
    image_path_list, video_path_list, audio_path_list = [], [], []
    for st in split_text:
        if st.startswith('<Image>'):
            pattern = r'Image>(.*?)<\/Image'
            matches = re.findall(pattern, text)
            for m in matches:
                image_path_list.append(m)
        elif st.startswith('<Audio>'):
            pattern = r'Audio>(.*?)<\/Audio'
            matches = re.findall(pattern, text)
            for m in matches:
                audio_path_list.append(m)
        elif st.startswith('<Video>'):
            pattern = r'Video>(.*?)<\/Video'
            matches = re.findall(pattern, text)
            for m in matches:
                video_path_list.append(m)
        else:
            res_text += st
    text = res_text
    if image_path is not None:
        text += f'<img src="./file={image_path}" style="display: inline-block;width: 250px;max-height: 400px;"><br>'
        outputs = f'<Image>{image_path}</Image> ' + outputs
    if len(image_path_list) > 0:
        for i in image_path_list:
            text += f'<img src="./file={i}" style="display: inline-block;width: 250px;max-height: 400px;"><br>'
            outputs = f'<Image>{i}</Image> ' + outputs
    if video_path is not None:
        text += f' <video controls playsinline width="500" style="display: inline-block;"  src="./file={video_path}"></video><br>'
        outputs = f'<Video>{video_path}</Video> ' + outputs
    if len(video_path_list) > 0:
        for i in video_path_list:
            text += f' <video controls playsinline width="500" style="display: inline-block;"  src="./file={i}"></video><br>'
            outputs = f'<Video>{i}</Video> ' + outputs
    if audio_path is not None:
        text += f'<audio controls playsinline><source src="./file={audio_path}" type="audio/wav"></audio><br>'
        outputs = f'<Audio>{audio_path}</Audio> ' + outputs
    if len(audio_path_list) > 0:
        for i in audio_path_list:
            text += f'<audio controls playsinline><source src="./file={i}" type="audio/wav"></audio><br>'
            outputs = f'<Audio>{i}</Audio> ' + outputs
    # text = text[::-1].replace(">rb<", "", 1)[::-1]
    text = text[:-len("<br>")].rstrip() if text.endswith("<br>") else text
    return text, outputs


def save_image_to_local(image: Image.Image):
    # TODO: Update so the url path is used, to prevent repeat saving.
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image.save(filename)
    return filename


def save_video_to_local(video):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    writer = imageio.get_writer(filename, format='FFMPEG', fps=8)
    for frame in video:
        writer.append_data(frame)
    writer.close()
    return filename


def save_audio_to_local(audio):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.wav')
    scipy.io.wavfile.write(filename, rate=16000, data=audio)
    return filename


def parse_reponse(model_outputs):
    response = ''
    text_outputs = []
    for output_i, p in enumerate(model_outputs):
        if isinstance(p, str):
            response += p
            response += '<br>'
            text_outputs.append(p)
        elif 'img' in p.keys():
            _temp_output = ''
            for m in p['img']:
                if isinstance(m, str):
                    response += m.replace(' '.join([f'[IMG{i}]' for i in range(args['num_gen_img_tokens'])]), '')
                    response += '<br>'
                    _temp_output += m.replace(' '.join([f'[IMG{i}]' for i in range(args['num_gen_img_tokens'])]), '')
                else:
                    filename = save_image_to_local(m[0])
                    print(filename)
                    _temp_output = f'<Image>{filename}</Image> ' + _temp_output
                    response += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
            text_outputs.append(_temp_output)
        elif 'vid' in p.keys():
            _temp_output = ''
            for idx, m in enumerate(p['vid']):
                if isinstance(m, str):
                    response += m.replace(' '.join([f'[VID{i}]' for i in range(args['num_gen_video_tokens'])]), '')
                    response += '<br>'
                    _temp_output += m.replace(' '.join([f'[VID{i}]' for i in range(args['num_gen_video_tokens'])]), '')
                else:
                    filename = save_video_to_local(m)
                    print(filename)
                    _temp_output = f'<Video>{filename}</Video> ' + _temp_output
                    response += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'
            text_outputs.append(_temp_output)
        elif 'aud' in p.keys():
            _temp_output = ''
            for idx, m in enumerate(p['aud']):
                if isinstance(m, str):
                    response += m.replace(' '.join([f'[AUD{i}]' for i in range(args['num_gen_audio_tokens'])]), '')
                    response += '<br>'
                    _temp_output += m.replace(' '.join([f'[AUD{i}]' for i in range(args['num_gen_audio_tokens'])]), '')
                else:
                    filename = save_audio_to_local(m)
                    print(filename)
                    _temp_output = f'<Audio>{filename}</Audio> ' + _temp_output
                    response += f'<audio controls playsinline><source src="./file={filename}" type="audio/wav"></audio>'
            text_outputs.append(_temp_output)
        else:
            pass
    response = response[:-len("<br>")].rstrip() if response.endswith("<br>") else response
    return response, text_outputs


def re_predict(
        prompt_input,
        image_path,
        audio_path,
        video_path,
        # thermal_path,
        chatbot,
        # max_length,
        top_p,
        temperature,
        history,
        modality_cache,
        guidance_scale_for_img,
        num_inference_steps_for_img,
        guidance_scale_for_vid,
        num_inference_steps_for_vid,
        num_frames,
        guidance_scale_for_aud,
        num_inference_steps_for_aud,
        audio_length_in_s
):
    # drop the latest query and answers and generate again

    q, a = history.pop()
    chatbot.pop()
    return predict(q, image_path, audio_path, video_path, chatbot, top_p,
                   temperature, history, modality_cache, guidance_scale_for_img, num_inference_steps_for_img,
                   guidance_scale_for_vid, num_inference_steps_for_vid, num_frames,
                   guidance_scale_for_aud, num_inference_steps_for_aud, audio_length_in_s)


def predict(
        prompt_input,
        image_path,
        audio_path,
        video_path,
        chatbot,
        top_p,
        temperature,
        history,
        modality_cache,
        guidance_scale_for_img,
        num_inference_steps_for_img,
        guidance_scale_for_vid,
        num_inference_steps_for_vid,
        num_frames,
        guidance_scale_for_aud,
        num_inference_steps_for_aud,
        audio_length_in_s
):
    # prepare the prompt
    prompt_text = ''

    if len(history) == 0:
        prompt_text += '### Human: '
        if image_path is not None:
            prompt_text += f'<Image>{image_path}</Image> '
        if audio_path is not None:
            prompt_text += f'<Audio>{audio_path}</Audio> '
        if video_path is not None:
            prompt_text += f'<Video>{video_path}</Video> '
        prompt_text += f' {prompt_input}'
    else:
        for idx, (q, a) in enumerate(history):
            if idx == 0:
                prompt_text += f'### Human: {q}\n### Assistant: {a}\n###'
            else:
                prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
        prompt_text += ' Human: '
        if image_path is not None:
            prompt_text += f'<Image>{image_path}</Image> '
        if audio_path is not None:
            prompt_text += f'<Audio>{audio_path}</Audio> '
        if video_path is not None:
            prompt_text += f'<Video>{video_path}</Video> '
        prompt_text += f' {prompt_input}'
    print('prompt_text: ', prompt_text)
    print('image_path: ', image_path)
    print('audio_path: ', audio_path)
    print('video_path: ', video_path)
    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        # 'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache,
        'filter_value': filter_value, 'min_word_tokens': min_word_tokens,
        'gen_scale_factor': gen_scale_factor, 'max_num_imgs': max_num_imgs,
        'stops_id': stops_id,
        'load_sd': load_sd,
        'generator': generator,
        'guidance_scale_for_img': guidance_scale_for_img,
        'num_inference_steps_for_img': num_inference_steps_for_img,

        'guidance_scale_for_vid': guidance_scale_for_vid,
        'num_inference_steps_for_vid': num_inference_steps_for_vid,
        'max_num_vids': max_num_vids,
        'height': height,
        'width': width,
        'num_frames': num_frames,

        'guidance_scale_for_aud': guidance_scale_for_aud,
        'num_inference_steps_for_aud': num_inference_steps_for_aud,
        'max_num_auds': max_num_auds,
        'audio_length_in_s': audio_length_in_s,
        'ENCOUNTERS': ENCOUNTERS,
    })
    response_chat, response_outputs = parse_reponse(response)
    print('text_outputs: ', response_outputs)
    user_chat, user_outputs = parse_text(prompt_input, image_path, video_path, audio_path)
    chatbot.append((user_chat, response_chat))
    history.append((user_outputs, ''.join(response_outputs).replace('\n###', '')))
    return chatbot, history, modality_cache, None, None, None,


def reset_user_input():
    return gr.update(value='')


def reset_dialog():
    return [], []


def reset_state():
    return None, None, None, None, [], [], []


def upload_image(conversation, chat_history, image_input):
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    input_image.save(image_input.name)  # Overwrite with smaller image.
    conversation += [(f'<img src="./file={image_input.name}" style="display: inline-block;">', "")]
    return conversation, chat_history + [input_image, ""]


def upload_image_video_audio(gr_image, gr_video, gr_audio, chatbot, history):
    if gr_image is not None:
        print(gr_image)
        chatbot.append(((gr_image.name,), None))
        history = history + [((gr_image,), None)]
    if gr_video is not None:
        print(gr_video)
        chatbot.append(((gr_video.name,), None))
        history = history + [((gr_video,), None)]
    if gr_audio is not None:
        print(gr_audio)
        chatbot.append(((gr_audio.name,), None))
        history = history + [((gr_audio,), None)]
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), chatbot, history


with gr.Blocks() as demo:

    gr.HTML("""
        <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; "><img src='./file=nextgpt.png' width="45" height="45" style="margin-right: 10px;">NExT-GPT</h1>
        <h3>This is the demo page of NExT-GPT, an any-to-any multimodal LLM that allows for seamless conversion and generation among text, image, video and audio!</h3>
        <h3>The current initial version of NExT-GPT, limited by the quantity of fine-tuning data and the quality of the base models, may generate some low-quality or hallucinated content. Please interpret the results with caution. We will continue to update the model to enhance its performance. Thank you for trying the demo! If you have any questions or feedback, feel free to contact us.</h3>
        <div style="display: flex;"><a href='https://next-gpt.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp  &nbsp  &nbsp <a href='https://github.com/NExT-GPT/NExT-GPT'><img src='https://img.shields.io/badge/Github-Code-blue'></a> &nbsp &nbsp  &nbsp  <a href='https://arxiv.org/pdf/2309.05519.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
        """)

    with gr.Row():
        with gr.Column(scale=0.7, min_width=500):
            with gr.Row():
                chatbot = gr.Chatbot(label='NExT-GPT Chatbot', avatar_images=((os.path.join(os.path.dirname(__file__), 'user.png')), (os.path.join(os.path.dirname(__file__), "bot.png")))).style(height=440)

            with gr.Tab("User Input"):
                with gr.Row(scale=3):
                    user_input = gr.Textbox(label="Text", placeholder="Key in something here...", lines=3)
                with gr.Row(scale=3):
                    with gr.Column(scale=1):
                        # image_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])
                        image_path = gr.Image(type="filepath", label="Image")  # .style(height=200)  # <PIL.Image.Image image mode=RGB size=512x512 at 0x7F6E06738D90>
                    with gr.Column(scale=1):
                        audio_path = gr.Audio(type='filepath')  #.style(height=200)
                    with gr.Column(scale=1):
                        video_path = gr.Video()  #.style(height=200) # , value=None, interactive=True
        with gr.Column(scale=0.3, min_width=300):
            with gr.Group():
                with gr.Accordion('Text Advanced Options', open=True):
                    top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0, 1, value=1.0, step=0.01, label="Temperature", interactive=True)
                with gr.Accordion('Image Advanced Options', open=True):
                    guidance_scale_for_img = gr.Slider(1, 10, value=7.5, step=0.5, label="Guidance scale",
                                                       interactive=True)
                    num_inference_steps_for_img = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                                                            interactive=True)
                with gr.Accordion('Video Advanced Options', open=False):
                    guidance_scale_for_vid = gr.Slider(1, 10, value=7.5, step=0.5, label="Guidance scale",
                                                       interactive=True)
                    num_inference_steps_for_vid = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                                                            interactive=True)
                    num_frames = gr.Slider(label='Number of frames', minimum=16, maximum=32, step=8, value=24,
                                           interactive=True,
                                           info='Note that the content of the video also changes when you change the number of frames.')
                with gr.Accordion('Audio Advanced Options', open=False):
                    guidance_scale_for_aud = gr.Slider(1, 10, value=7.5, step=0.5, label="Guidance scale",
                                                       interactive=True)
                    num_inference_steps_for_aud = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                                                            interactive=True)
                    audio_length_in_s = gr.Slider(1, 9, value=9, step=1, label="The audio length in seconds",
                                                  interactive=True)
            with gr.Tab("Operation"):
                with gr.Row(scale=1):
                    submitBtn = gr.Button(value="Submit & Run", variant="primary")
                with gr.Row(scale=1):
                    resubmitBtn = gr.Button("Rerun")
                with gr.Row(scale=1):
                    emptyBtn = gr.Button("Clear History")

    history = gr.State([])
    modality_cache = gr.State([])

    submitBtn.click(
        predict, [
            user_input,
            image_path,
            audio_path,
            video_path,
            chatbot,
            # max_length,
            top_p,
            temperature,
            history,
            modality_cache,
            guidance_scale_for_img,
            num_inference_steps_for_img,
            guidance_scale_for_vid,
            num_inference_steps_for_vid,
            num_frames,
            guidance_scale_for_aud,
            num_inference_steps_for_aud,
            audio_length_in_s
        ], [
            chatbot,
            history,
            modality_cache,
            image_path,
            audio_path,
            video_path
        ],
        show_progress=True
    )

    resubmitBtn.click(
        re_predict, [
            user_input,
            image_path,
            audio_path,
            video_path,
            chatbot,
            # max_length,
            top_p,
            temperature,
            history,
            modality_cache,
            guidance_scale_for_img,
            num_inference_steps_for_img,
            guidance_scale_for_vid,
            num_inference_steps_for_vid,
            num_frames,
            guidance_scale_for_aud,
            num_inference_steps_for_aud,
            audio_length_in_s
        ], [
            chatbot,
            history,
            modality_cache,
            image_path,
            audio_path,
            video_path
        ],
        show_progress=True
    )

    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[
        image_path,
        audio_path,
        video_path,
        chatbot,
        history,
        modality_cache
    ], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=24004)
