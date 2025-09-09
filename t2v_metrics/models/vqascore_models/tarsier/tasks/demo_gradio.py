# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copy and modify from: https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/demo/demo.py

# import spaces # for deploying on huggingface ZeroGPU
from copy import deepcopy
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
from tools.conversation import Chat, conv_templates
from tasks.utils import load_model_and_processor, file_to_base64
from dataset.tarsier_datamodule import init_processor
import os
import torch

# huggingface-cli login

model_path = os.getenv("MODEL_PATH", "omni-research/Tarsier2-7b")
config_path = "configs/tarser2_default_config.yaml"
max_n_frames = int(os.getenv("MAX_N_FRAMES", 16))
debug = False
device = 'cuda' if not debug else 'cpu'

# ========================================
#             Model Initialization
# ========================================
def init_model():
    print("Start Initialization...")
    # if torch.cuda.is_available():
    if not debug:
        model, processor = load_model_and_processor(model_path, config_path)
    else:
        print(f"No Valid GPU! Lauch in debug mode!")
        processor = init_processor(model_path, config_path)
        model = None
    chat = Chat(model, processor, device, debug)c   
    print('Initialization Finished')
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_file):
    if chat_state is not None:
        chat_state.messages = []
    img_file = None
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_file


def upload_img(gr_img, gr_video, gr_gif, chat_state, num_frames):
    print("video, image or gif:", gr_video, gr_img, gr_gif)
    conv_type = ''
    if 'tarsier2-7b' in model_path.lower():
        conv_type = 'tarsier2-7b'
    # elif '7b' in model_path.lower():
    #     conv_type = 'tarsier-7b'
    # elif '13b' in model_path.lower():
    #     conv_type = 'tarsier-13b'
    # elif '34b' in model_path.lower():
    #     conv_type = 'tarsier-34b'
    else:
        raise ValueError(f"Unknow model: {model_path}")
    chat_state = deepcopy(conv_templates[conv_type])

    if gr_img is None and gr_video is None and gr_gif is None:
        return None, None, None, gr.update(interactive=True), gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, None, None
    if gr_video or gr_img or gr_gif:
        for img_file in [gr_video, gr_img, gr_gif]:
            if img_file is not None:
                break
        chat_state.messages.append([chat_state.roles[0], {"type": "video", "text": img_file}])
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_file


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state = chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state

# @spaces.GPU(duration=120) # for deploying on huggingface ZeroGPU
def gradio_answer(chatbot, chat_state, img_file, top_p, temperature, n_frames=None):
    llm_message, chat_state = chat.answer(conv=chat_state, n_frames=n_frames, max_new_tokens=256, num_beams=1, temperature=temperature, top_p=top_p)
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

logo_b64 = file_to_base64("assets/figures/tarsier_logo.jpg")
title = f"""<center><a href="https://github.com/bytedance/tarsier"><img src="data:image/jpeg;base64,{logo_b64}" alt="Tarsier" border="0" style="margin: 0 auto; height: 140px;" /></a></center>"""
description ="""<center><p><a href='https://github.com/bytedance/tarsier'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p></center>
"""


with gr.Blocks(title="Tarsier",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload", height=360)
                with gr.Tab("Image", elem_id='image_tab'):
                    up_image = gr.Image(type="filepath", interactive=True, elem_id="image_upload", height=360)
                with gr.Tab("GIF", elem_id='gif_tab'):
                    up_gif = gr.File(type="filepath", file_count="single", file_types=[".gif"], interactive=True, elem_id="gif_upload", height=360)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            # num_beams = gr.Slider(
            #     minimum=1,
            #     maximum=10,
            #     value=1,
            #     step=1,
            #     interactive=True,
            #     label="beam search numbers)",
            # )
            
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Top_p",
            )
            
            num_frames = gr.Slider(
                minimum=4,
                maximum=16,
                value=16,
                step=2,
                interactive=True,
                label="#Frames",
            )
        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State()
            img_file = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False, container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")
    
    chat = init_model()
    upload_button.click(upload_img, [up_image, up_video, up_gif, chat_state, num_frames], [up_image, up_video, up_gif, text_input, upload_button, chat_state, img_file])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_file, top_p, temperature, num_frames], [chatbot, chat_state]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_file, top_p, temperature, num_frames], [chatbot, chat_state]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_file], [chatbot, up_image, up_video, up_gif, text_input, upload_button, chat_state, img_file], queue=False)


demo.launch()
# demo.launch(server_name="0.0.0.0", server_port=11451)