# !git clone https://huggingface.co/spaces/openai/whisper
# %cd whisper
# !pip install -r requirements.txt
# !pip install gradio
# install youtube-dl
# !pip install youtube-dl

# Goal is also to understand and summarize foreign language videos
# check on a huge scale if certain keywords appear in foreign media, startup idea

import youtube_dl
import googletrans
import os
from typing import Tuple
# os.system("pip install git+https://github.com/openai/whisper.git")

import whisper

#from share_btn import community_icon_html, loading_icon_html, share_js

model = whisper.load_model("small")
content_dir = "content"



def inference(audio: str) -> Tuple[str, str]:
    # model = whisper.load_model("small")
    result = model.transcribe(audio)
    # result also contains segments
    return result["text"], result['language']

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
    return f"{content_dir}/{identifier}.wav"

def download_audio_from_reddit(url: str) -> str:
    """
    url: reddit url

    returns: filename
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{content_dir}/reddit.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return f"{content_dir}/reddit.wav"

def transcribe(identifier: str) -> Tuple[str, str]:
    if os.path.exists(f"{content_dir}/{identifier}.wav"):
        filename = f"{content_dir}/{identifier}.wav"
    else:
        filename = download_audio(identifier=identifier)
    return inference(filename)

def translate(text: str, src_language: str) -> str:
    # translate with google translate
    translator = googletrans.Translator()
    text = translator.translate(text, dest="en", src=src_language).text
    return text

def summarize(text: str) -> str:
    # idealy you should summarize with openai api
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    # summarizing with t5-small does not make much sense. results are not good
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # input  text and return the output of the model after sampling
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=1000, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0])

if __name__ == "__main__":
    # commandline interface for the inference function
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-i","--identifier", help="youtube identifier", default=None,  type=str)
    parser.add_argument("-f","--filename", help="path to audio file", default=None,  type=str)
    parser.add_argument("-t","--translate", help="translate the text into english", default=True, type=bool)
    parser.add_argument("-s","--summarize", help="summarize the text", default=False, type=bool)

    args = parser.parse_args()
    # only one of identifier or filename can not be None
    assert not (args.identifier and args.filename), "Please provide only one of identifier or filename"




    if args.identifier:
        text, src_language = transcribe(identifier=args.identifier)
        filename_without_extension = args.identifier
    elif args.filename:
        filename_without_extension = args.filename.split(".")[0]
        # try conversion to wav file
        if not args.filename.endswith(".wav"):
            os.system(f"ffmpeg -i {content_dir}/{args.filename} {content_dir}/{filename_without_extension}.wav")
            # check if file with the filename exists
            if not os.path.exists(f"{content_dir}/{filename_without_extension}.wav"):
                print("Please provide a valid filename")
                exit()
        text, src_language = inference(f"{content_dir}/{args.filename}")
        
    else:
        print("Please provide an identifier or a filename parameter:  --identifier: IYgZS2EvnLI")
        exit()

    if text and filename_without_extension:
        if args.translate and not src_language == "en":
            text = translate(text, src_language=src_language)
        if args.summarize:
            text = summarize(text)
        with open(f"{content_dir}/{filename_without_extension}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    # todo, offer conversion to wav file and summarization

