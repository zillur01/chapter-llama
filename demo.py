import os
import tempfile
from pathlib import Path
from urllib.request import getproxies

import gradio as gr
from llama_cookbook.inference.model_utils import load_model as load_model_llamarecipes
from llama_cookbook.inference.model_utils import load_peft_model
from transformers import AutoTokenizer

from src.data.single_video import SingleVideo
from src.data.utils_asr import PromptASR
from src.models.llama_inference import inference
from src.test.vidchapters import get_chapters
from src.utils import RankedLogger
from tools.download.models import download_model

log = RankedLogger(__name__, rank_zero_only=True)

# Set up proxies
proxies = getproxies()
os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = proxies["http"]
os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = proxies["https"]
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"

# Global variables to store loaded models
base_model = None
tokenizer = None
current_peft_model = None
inference_model = None

LLAMA_CKPT_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def load_base_model():
    """Load the base Llama model and tokenizer once at startup."""
    global base_model, tokenizer

    if base_model is None:
        log.info(f"Loading base model: {LLAMA_CKPT_PATH}")
        base_model = load_model_llamarecipes(
            model_name=LLAMA_CKPT_PATH,
            device_map="auto",
            quantization=None,
            use_fast_kernels=True,
        )
        base_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(LLAMA_CKPT_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        log.info("Base model loaded successfully")


class FastLlamaInference:
    def __init__(
        self,
        model,
        add_special_tokens: bool = True,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 50,
        use_cache: bool = True,
        max_padding_length: int = None,
        do_sample: bool = False,
        min_length: int = None,
        repetition_penalty: float = 1.0,
        length_penalty: int = 1,
        max_prompt_tokens: int = 35_000,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.max_padding_length = max_padding_length
        self.do_sample = do_sample
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.max_prompt_tokens = max_prompt_tokens

    def __call__(self, prompt: str, **kwargs):
        # Create a dict of default parameters from instance attributes
        params = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "add_special_tokens": self.add_special_tokens,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_cache": self.use_cache,
            "max_padding_length": self.max_padding_length,
            "do_sample": self.do_sample,
            "min_length": self.min_length,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "max_prompt_tokens": self.max_prompt_tokens,
        }

        # Update with any overrides passed in kwargs
        params.update(kwargs)

        return inference(**params)


def load_peft(model_name: str = "asr-10k"):
    """Load or switch PEFT model while reusing the base model."""
    global base_model, current_peft_model, inference_model

    # First make sure the base model is loaded
    if base_model is None:
        load_base_model()

    # Only load a new PEFT model if it's different from the current one
    if current_peft_model != model_name:
        log.info(f"Loading PEFT model: {model_name}")
        model_path = download_model(model_name)

        if not Path(model_path).exists():
            log.warning(f"PEFT model does not exist at {model_path}")
            return False

        # Apply the PEFT model to the base model
        peft_model = load_peft_model(base_model, model_path)

        peft_model.eval()

        # Create the inference wrapper
        inference_model = FastLlamaInference(model=peft_model)
        current_peft_model = model_name

        log.info(f"PEFT model {model_name} loaded successfully")
        return True

    # Model already loaded
    return True


def process_video(video_file, model_name: str = "asr-10k", do_sample: bool = False):
    """Process a video file and generate chapters."""
    if video_file is None:
        return "Please upload a video file."

    progress = gr.Progress()
    progress(0, desc="Starting...")

    # Load the PEFT model
    progress(0.1, desc=f"Loading LoRA parameters from {model_name}...")
    if not load_peft(model_name):
        return "Failed to load model. Please try again."

    # Create a temporary directory to save the uploaded video
    with tempfile.TemporaryDirectory() as temp_dir:
        progress(0.2, desc="Processing uploaded video...")
        temp_video_path = Path(temp_dir) / "temp_video.mp4"
        # Save the uploaded file
        with open(temp_video_path, "wb") as f:
            f.write(video_file)

        # Process the video
        progress(0.3, desc="Extracting ASR transcript...")
        single_video = SingleVideo(temp_video_path)
        progress(0.4, desc="Creating prompt...")
        prompt = PromptASR(chapters=single_video)

        vid_id = single_video.video_ids[0]
        progress(0.5, desc="Creating prompt...")
        prompt = prompt.get_prompt_test(vid_id)

        transcript = single_video.get_asr(vid_id)
        prompt = prompt + transcript

        progress(0.6, desc="Generating chapters with Chapter-Llama...")
        _, chapters = get_chapters(
            inference_model,
            prompt,
            max_new_tokens=1024,
            do_sample=do_sample,
            vid_id=vid_id,
        )

        # Format the output
        progress(0.9, desc="Formatting results...")
        output = ""
        for timestamp, text in chapters.items():
            output += f"{timestamp}: {text}\n"

        progress(1.0, desc="Complete!")
        return output


# Create the Gradio interface
with gr.Blocks(title="Chapter-Llama") as demo:
    gr.Markdown("# Chapter-Llama")
    gr.Markdown("## Chaptering in Hour-Long Videos with LLMs")
    gr.Markdown("Upload a video file to generate chapters automatically.")
    gr.Markdown(
        """
        This demo is currently using only the audio data (ASR), without frame information. 
        We will add audio+captions functionality in the near future, which will improve 
        chapter generation by incorporating visual content.
        
        - GitHub: [https://github.com/lucas-ventura/chapter-llama](https://github.com/lucas-ventura/chapter-llama)
        - Website: [https://imagine.enpc.fr/~lucas.ventura/chapter-llama/](https://imagine.enpc.fr/~lucas.ventura/chapter-llama/)
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="Upload Video", file_types=["video", "audio"], type="binary"
            )
            model_dropdown = gr.Dropdown(
                choices=["asr-10k", "asr-1k"],
                value="asr-10k",
                label="Select Model",
            )
            do_sample = gr.Checkbox(
                label="Use random sampling", value=False, interactive=True
            )
            submit_btn = gr.Button("Generate Chapters")

        with gr.Column():
            status_area = gr.Markdown("**Status:** Ready to process video")
            output_text = gr.Textbox(
                label="Generated Chapters", lines=10, interactive=False
            )

    def update_status_and_process(video_file, model_name, do_sample):
        if video_file is None:
            return "**Status:** No video uploaded", "Please upload a video file."
        else:
            return "**Status:** Processing video...", process_video(
                video_file, model_name, do_sample
            )

    # Load the base model at startup
    load_base_model()

    submit_btn.click(
        fn=update_status_and_process,
        inputs=[video_input, model_dropdown, do_sample],
        outputs=[status_area, output_text],
    )


if __name__ == "__main__":
    # Launch the Gradio app
    demo.launch(share=True)
