import gradio as gr
from urllib.parse import urlparse
import requests
import time
import os

from utils.gradio_helpers import parse_outputs, process_outputs

inputs = []
inputs.append(gr.Image(
    label="Image", type="filepath"
))

inputs.append(gr.Textbox(
    label="Prompt", info='''Prompt'''
))

inputs.append(gr.Textbox(
    label="Negative Prompt", info='''Negative Prompt'''
))

inputs.append(gr.Number(
    label="Scale Factor", info='''Scale factor''', value=2
))

inputs.append(gr.Slider(
    label="Dynamic", info='''HDR, try from 3 - 9''', value=6,
    minimum=1, maximum=50
))

inputs.append(gr.Number(
    label="Creativity", info='''Creativity, try from 0.3 - 0.9''', value=0.35
))

inputs.append(gr.Number(
    label="Resemblance", info='''Resemblance, try from 0.3 - 1.6''', value=0.6
))

inputs.append(gr.Dropdown(
    choices=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256], label="tiling_width", info='''Fractality, set lower tile width for a high Fractality''', value="112"
))

inputs.append(gr.Dropdown(
    choices=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256], label="tiling_height", info='''Fractality, set lower tile height for a high Fractality''', value="144"
))

inputs.append(gr.Dropdown(
    choices=['epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]', 'juggernaut_reborn.safetensors [338b85bc4f]', 'flat2DAnimerge_v45Sharp.safetensors'], label="sd_model", info='''Stable Diffusion model checkpoint''', value="juggernaut_reborn.safetensors [338b85bc4f]"
))

inputs.append(gr.Dropdown(
    choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras', 'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential', 'DPM fast', 'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras', 'Restart', 'DDIM', 'PLMS', 'UniPC'], label="scheduler", info='''scheduler''', value="DPM++ 3M SDE Karras"
))

inputs.append(gr.Slider(
    label="Num Inference Steps", info='''Number of denoising steps''', value=18,
    minimum=1, maximum=100, step=1,
))

inputs.append(gr.Number(
    label="Seed", info='''Random seed. Leave blank to randomize the seed''', value=1337
))

inputs.append(gr.Checkbox(
    label="Downscaling", info='''Downscale the image before upscaling. Can improve quality and speed for images with high resolution but lower quality''', value=False
))

inputs.append(gr.Number(
    label="Downscaling Resolution", info='''Downscaling resolution''', value=768
))

inputs.append(gr.Textbox(
    label="Lora Links", info='''Link to a lora file you want to use in your upscaling. Multiple links possible, seperated by comma'''
))

inputs.append(gr.Textbox(
    label="Custom Sd Model", info='''Link to a custom safetensors checkpoint file you want to use in your upscaling. Will overwrite sd_model checkpoint.'''
))

names = ['image', 'prompt', 'negative_prompt', 'scale_factor', 'dynamic', 'creativity', 'resemblance', 'tiling_width', 'tiling_height', 'sd_model', 'scheduler', 'num_inference_steps', 'seed', 'downscaling', 'downscaling_resolution', 'lora_links', 'custom_sd_model']

outputs = []
outputs.append(gr.Image())

expected_outputs = len(outputs)
def predict(request: gr.Request, *args, progress=gr.Progress(track_tqdm=True)):
    headers = {'Content-Type': 'application/json'}

    payload = {"input": {}}
    
    
    base_url = "http://0.0.0.0:7860"
    for i, key in enumerate(names):
        value = args[i]
        if value and (os.path.exists(str(value))):
            value = f"{base_url}/file=" + value
        if value is not None and value != "":
            payload["input"][key] = value

    response = requests.post("http://0.0.0.0:5000/predictions", headers=headers, json=payload)

    
    if response.status_code == 201:
        follow_up_url = response.json()["urls"]["get"]
        response = requests.get(follow_up_url, headers=headers)
        while response.json()["status"] != "succeeded":
            if response.json()["status"] == "failed":
                raise gr.Error("The submission failed!")
            response = requests.get(follow_up_url, headers=headers)
            time.sleep(1)
    if response.status_code == 200:
        json_response = response.json()
        #If the output component is JSON return the entire output response 
        if(outputs[0].get_config()["name"] == "json"):
            return json_response["output"]
        predict_outputs = parse_outputs(json_response["output"])
        processed_outputs = process_outputs(predict_outputs)
        difference_outputs = expected_outputs - len(processed_outputs)
        # If less outputs than expected, hide the extra ones
        if difference_outputs > 0:
            extra_outputs = [gr.update(visible=False)] * difference_outputs
            processed_outputs.extend(extra_outputs)
        # If more outputs than expected, cap the outputs to the expected number
        elif difference_outputs < 0:
            processed_outputs = processed_outputs[:difference_outputs]
        
        return tuple(processed_outputs) if len(processed_outputs) > 1 else processed_outputs[0]
    else:
        if(response.status_code == 409):
            raise gr.Error(f"Sorry, the Cog image is still processing. Try again in a bit.")
        raise gr.Error(f"The submission failed! Error: {response.status_code}")

title = "Demo for clarity-upscaler cog image by philz1337x"
model_description = "High resolution image Upscaler and Enhancer. Use at ClarityAI.cc. A free Magnific alternative. Twitter/X: @philz1337x"

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=model_description,
    allow_flagging="never",
)
app.launch(share=True)

