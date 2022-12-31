# !git clone https://huggingface.co/spaces/openai/whisper
# %cd whisper
# !pip install -r requirements.txt
# !pip install gradio
# install youtube-dl
# !pip install youtube-dl

# Goal is also to understand and summarize foreign language videos
# check on a huge scale if certain keywords appear in foreign media, startup idea

import youtube_dl

import os
# os.system("pip install git+https://github.com/openai/whisper.git")
#import gradio as gr
import whisper

#from share_btn import community_icon_html, loading_icon_html, share_js

model = whisper.load_model("small")
content_dir = "content"


        
def inference_trim(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _, probs = model.detect_language(mel)
    
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    
    print(result.text)
    return result.text

def inference(audio):
    # model = whisper.load_model("small")
    result = model.transcribe(audio)
    return result["text"]

# download the audio file from youtube
def download_audio(identifier: str) -> str:
    """
    identifier: youtube identifier string

    returns: filename
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{content_dir}/{identifier}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([identifier])
    return f"{identifier}.wav"

def transcribe(identifier: str) -> str:
    if os.path.exists(f"{content_dir}/{identifier}.wav"):
        filename = f"{content_dir}/{identifier}.wav"
    else:
        filename = download_audio(identifier=identifier)
    text = inference(filename)
    # summarize with gpt
    # Create an update from Youtube
    return text

# example identifier IYgZS2EvnLI

if __name__=="__main__":
    
    # create content folder if it doesn't exist
    os.path.exists(content_dir) or os.mkdir("content")

    identifier = "IYgZS2EvnLI"
    text = transcribe(identifier=identifier)
    # write text into file with name identifier.txt
    with open(f"{content_dir}/{identifier}.txt", "w") as f:
        f.write(text)

    # todo: summarize with gpt
