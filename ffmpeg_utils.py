import subprocess

FFMPEG = "ffmpeg"

class FFmpegError(RuntimeError):
    pass

def run_ffmpeg(args: list[str]) -> None:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="ignore"))

def extract_audio_wav(video_path: str, out_wav: str, sr: int = 16000, mono: bool = True) -> str:
    args = ["-y", "-i", video_path]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sr), out_wav]
    run_ffmpeg(args)
    return out_wav

def ffmpeg_extract(in_wav: str, out_wav: str, start: float, end: float) -> None:
    duration = max(0.1, end - start)
    run_ffmpeg(["-y", "-ss", f"{start}", "-t", f"{duration}", "-i", in_wav, out_wav])

def mux_replace_audio(video_path: str, new_audio_wav: str, out_video: str) -> str:
    run_ffmpeg([
        "-y", "-i", video_path, "-i", new_audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", out_video,
    ])
    return out_video
