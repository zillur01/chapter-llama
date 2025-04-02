from pathlib import Path

from lutils import openf, writef

from src.data.chapters import sec_to_hms
from tools.extract.asr import ASRProcessor


class SingleVideo:
    """
    A simplified implementation of the src.data.chapters.Chapters interface for single video inference.

    This class mimics the behavior of the ChaptersASR class but is designed to work with
    a single video file rather than a dataset. It provides the necessary methods
    required by the PromptASR class for generating chapter timestamps and titles.

    Note: This class is intended for inference only and should not be used for
    training or evaluation purposes.
    """

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.video_ids = [video_path.stem]
        assert video_path.exists(), f"Video file {video_path} not found"
        self.asr, self.duration = get_asr(video_path, overwrite=True)

    def __len__(self):
        return len(self.video_ids)

    def __iter__(self):
        return iter(self.video_ids)

    def __contains__(self, vid_id):
        return vid_id in self.video_ids

    def get_duration(self, vid_id, hms=False):
        assert vid_id == self.video_ids[0], f"Invalid video ID: {vid_id}"
        if hms:
            return sec_to_hms(self.duration)
        return self.duration

    def get_asr(self, vid_id):
        assert vid_id == self.video_ids[0], f"Invalid video ID: {vid_id}"
        return self.asr


def get_asr(video_path: Path, overwrite=False):
    output_dir = Path(f"outputs/inference/{video_path.stem}")
    asr_output = output_dir / "asr.txt"
    duration_output = output_dir / "duration.txt"
    if asr_output.exists() and duration_output.exists() and not overwrite:
        asr = openf(asr_output)
        asr = "\n".join(asr) + "\n"

        duration = openf(duration_output)
        assert isinstance(duration, list) and len(duration) == 1, (
            f"Duration is not a list of length 1: {duration}"
        )
        duration = float(duration[0])
        assert duration > 0, f"Duration is not positive: {duration}"
        return asr, duration

    print(f"\n=== 🎙️ Processing ASR for {video_path} ===")
    asr_processor = ASRProcessor()
    asr, duration = asr_processor.get_asr(video_path)
    print(f"=== ✅ ASR processing complete for {video_path} ===\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    writef(asr_output, asr)
    writef(duration_output, str(duration))
    return asr, duration
