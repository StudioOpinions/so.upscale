import gradio as gr
from PIL import Image
from diffusers import AutoPipelineForImage2Image
import torch


# Load the upscaler pipeline lazily so that the model is only
# downloaded and moved to the appropriate device on first use.
pipe = None


def load_pipeline():
    """Load the clarity upscaler pipeline if it hasn't been loaded yet."""
    global pipe
    if pipe is None:
        model_id = "Philz1337x/clarity-upscaler"
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)


def predict(
    image,
    prompt,
    negative_prompt,
    scale_factor,
    dynamic,
    creativity,
    resemblance,
    tiling_width,
    tiling_height,
    sd_model,
    scheduler,
    num_inference_steps,
    seed,
    downscaling,
    downscaling_resolution,
    lora_links,
    custom_sd_model,
):
    """Run local inference using the clarity upscaler model.

    Most UI parameters are accepted for compatibility with the original app
    but only the essential ones are used by the pipeline. Unused parameters
    are ignored.
    """

    load_pipeline()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    init_image = Image.open(image).convert("RGB")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        num_inference_steps=int(num_inference_steps),
        generator=generator,
    )

    upscaled = result.images[0]
    if scale_factor and scale_factor != 1:
        new_size = (
            int(upscaled.width * float(scale_factor)),
            int(upscaled.height * float(scale_factor)),
        )
        upscaled = upscaled.resize(new_size, Image.LANCZOS)

    return upscaled


inputs = []
inputs.append(gr.Image(label="Image", type="filepath"))
inputs.append(gr.Textbox(label="Prompt", info="""Prompt"""))
inputs.append(gr.Textbox(label="Negative Prompt", info="""Negative Prompt"""))
inputs.append(gr.Number(label="Scale Factor", info="""Scale factor""", value=2))
inputs.append(
    gr.Slider(
        label="Dynamic",
        info="""HDR, try from 3 - 9""",
        value=6,
        minimum=1,
        maximum=50,
    )
)
inputs.append(
    gr.Number(
        label="Creativity",
        info="""Creativity, try from 0.3 - 0.9""",
        value=0.35,
    )
)
inputs.append(
    gr.Number(
        label="Resemblance",
        info="""Resemblance, try from 0.3 - 1.6""",
        value=0.6,
    )
)
inputs.append(
    gr.Dropdown(
        choices=[
            16,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            176,
            192,
            208,
            224,
            240,
            256,
        ],
        label="tiling_width",
        info="""Fractality, set lower tile width for a high Fractality""",
        value="112",
    )
)
inputs.append(
    gr.Dropdown(
        choices=[
            16,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            176,
            192,
            208,
            224,
            240,
            256,
        ],
        label="tiling_height",
        info="""Fractality, set lower tile height for a high Fractality""",
        value="144",
    )
)
inputs.append(
    gr.Dropdown(
        choices=[
            "epicrealism_naturalSinRC1VAE.safetensors [84d76a0328]",
            "juggernaut_reborn.safetensors [338b85bc4f]",
            "flat2DAnimeRge_v45Sharp.safetensors",
        ],
        label="sd_model",
        info="""Stable Diffusion model checkpoint""",
        value="juggernaut_reborn.safetensors [338b85bc4f]",
    )
)
inputs.append(
    gr.Dropdown(
        choices=[
            "DPM++ 2M Karras",
            "DPM++ SDE Karras",
            "DPM++ 2M SDE Exponential",
            "DPM++ 2M SDE Karras",
            "Euler a",
            "Euler",
            "LMS",
            "Heun",
            "DPM2",
            "DPM2 a",
            "DPM++ 2S a",
            "DPM++ 2M",
            "DPM++ SDE",
            "DPM++ 2M SDE",
            "DPM++ 2M SDE Heun",
            "DPM++ 2M SDE Heun Karras",
            "DPM++ 2M SDE Heun Exponential",
            "DPM++ 3M SDE",
            "DPM++ 3M SDE Karras",
            "DPM++ 3M SDE Exponential",
            "DPM fast",
            "DPM adaptive",
            "LMS Karras",
            "DPM2 Karras",
            "DPM2 a Karras",
            "DPM++ 2S a Karras",
            "Restart",
            "DDIM",
            "PLMS",
            "UniPC",
        ],
        label="scheduler",
        info="""scheduler""",
        value="DPM++ 3M SDE Karras",
    )
)
inputs.append(
    gr.Slider(
        label="Num Inference Steps",
        info="""Number of denoising steps""",
        value=18,
        minimum=1,
        maximum=100,
        step=1,
    )
)
inputs.append(
    gr.Number(
        label="Seed",
        info="""Random seed. Leave blank to randomize the seed""",
        value=1337,
    )
)
inputs.append(
    gr.Checkbox(
        label="Downscaling",
        info="""Downscale the image before upscaling. Can improve quality and speed for images with high resolution but lower quality""",
        value=False,
    )
)
inputs.append(
    gr.Number(
        label="Downscaling Resolution",
        info="""Downscaling resolution""",
        value=768,
    )
)
inputs.append(
    gr.Textbox(
        label="Lora Links",
        info="""Link to a lora file you want to use in your upscaling. Multiple links possible, seperated by comma""",
    )
)
inputs.append(
    gr.Textbox(
        label="Custom Sd Model",
        info="""Link to a custom safetensors checkpoint file you want to use in your upscaling. Will overwrite sd_model checkpoint.""",
    )
)


outputs = [gr.Image()]


title = "Demo for clarity-upscaler by philz1337x"
model_description = (
    "High resolution image Upscaler and Enhancer. Use at ClarityAI.cc. "
    "A free Magnific alternative. Twitter/X: @philz1337x"
)

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=model_description,
    allow_flagging="never",
)


if __name__ == "__main__":
    app.launch()

