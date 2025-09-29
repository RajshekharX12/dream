import os
from stt import transcribe
from translate_hi import build_translator, translate_segments
from voice_clone import VoiceCloner
from mux import compose_dubbed_track
from ffmpeg_utils import extract_audio_wav, ffmpeg_extract

VIDEO = "sample.mp4"  # put a short English video here
workdir = "_work"
os.makedirs(workdir, exist_ok=True)

wav16k = extract_audio_wav(VIDEO, os.path.join(workdir, "original.wav"), sr=16000)
segments = transcribe(wav16k)
translator = build_translator()
segments_hi = translate_segments(segments, translator)

ref_path = os.path.join(workdir, "ref.wav")
if segments:
    start = max(0.0, segments[0]['start'])
    end = start + min(8.0, max(6.0, segments[0]['end'] - segments[0]['start']))
else:
    start, end = 0.0, 7.0
ffmpeg_extract(wav16k, ref_path, start, end)

cloner = VoiceCloner()
dub_wav = os.path.join(workdir, "dubbed.wav")
compose_dubbed_track(
    original_wav=wav16k,
    segments=segments_hi,
    ref_speaker_wav=ref_path,
    out_wav=dub_wav,
    cloner=cloner,
)
print("Dubbed track:", dub_wav)
