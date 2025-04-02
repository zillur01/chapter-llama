import tempfile
from pathlib import Path

import gradio as gr

from src.data.single_video import SingleVideo
from src.data.utils_asr import PromptASR
from src.models.llama_inference import LlamaInference
from src.test.vidchapters import get_chapters
from tools.download.models import download_model

# import os
# from urllib.request import getproxies

# proxies = getproxies()
# os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = proxies["http"]
# os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = proxies["https"]
# os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"

# Global variables to store loaded models
inference_model = None
current_model_name = None


def load_model(model_name: str = "asr-10k"):
    """Load the model if it's not already loaded or if a different model is requested."""
    global inference_model, current_model_name

    if inference_model is None or current_model_name != model_name:
        model_path = download_model(model_name)
        inference_model = LlamaInference(
            ckpt_path="meta-llama/Llama-3.1-8B-Instruct", peft_model=model_path
        )
        current_model_name = model_name


def process_video(video_file, model_name: str = "asr-10k"):
    """Process a video file and generate chapters."""
    if video_file is None:
        return "Please upload a video file."

    # Create a temporary directory to save the uploaded video
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = Path(temp_dir) / "temp_video.mp4"
        # Save the uploaded file
        with open(temp_video_path, "wb") as f:
            f.write(video_file)

        # Load the model if needed
        load_model(model_name)

        # Process the video
        single_video = SingleVideo(temp_video_path)
        prompt = PromptASR(chapters=single_video)

        vid_id = single_video.video_ids[0]
        prompt = prompt.get_prompt_test(vid_id)
        transcript = single_video.get_asr(vid_id)
        prompt = prompt + transcript

        _, chapters = get_chapters(
            inference_model,
            prompt,
            max_new_tokens=1024,
            do_sample=False,
            vid_id=vid_id,
        )

        # Format the output
        output = ""
        for timestamp, text in chapters.items():
            output += f"{timestamp}: {text}\n"
        return output


# Create the Gradio interface
with gr.Blocks(title="Video Chapter Generator") as demo:
    gr.Markdown("# Video Chapter Generator")
    gr.Markdown("Upload a video file to generate chapters automatically.")
    gr.Markdown(
        """
        ## Note
        This demo is currently using only the audio data (ASR), without frame information. 
        We will add audio+captions functionality in the near future, which will improve 
        chapter generation by incorporating visual content.
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="Upload Video", file_types=["video"], type="binary"
            )
            model_dropdown = gr.Dropdown(
                choices=["asr-10k", "asr-1k"],
                value="asr-10k",
                label="Select Model",
            )
            submit_btn = gr.Button("Generate Chapters")

        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Chapters", lines=10, interactive=False
            )

    submit_btn.click(
        fn=process_video, inputs=[video_input, model_dropdown], outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(share=True)
