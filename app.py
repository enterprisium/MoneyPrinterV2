!pip install --quiet g4f[all] --upgrade
!pip install --quiet google-generativeai
!pip install --quiet TTS
!pip install --quiet ipyplot
!pip install --quiet git+https://github.com/Zulko/moviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5
!pip install --quiet imageio==2.33
!pip install --quiet numpy==1.23
!apt install -qq imagemagick
!cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml



# choose from Gemini or G4F
LLM = "Gemini" #  @param ["Gemini", "G4F"]
# Choose from Segmind or Hercai
VIDEO_SOURCE = "Hercai" #  @param ["Segmind",  "Hercai"]
# Choose  from XTTS_V2 or Elevenlabs
TTS_PROVIDER = "XTTS_V2" #  @param ["Elevenvlabs",  "XTTS_V2"]
# two words like "en, hi, fr, are for XTTS_V2
LANGUAGE = "en" # @param ["en", "English", "hi", "Hindi", "Chinese", "Spanish", "Portuguese", "French", "German", "Japanese", "Arabic", "Korean", "Italian", "Dutch", "Turkish", "Polish", "Russian", "Czech"]
# single name voices are only for elevenlabs, all others are for XTTS_V2
VOICE = "Baldur Sanjin" # @param ["Sarah", "Laura", "Charlie", "George", "Callum", "Liam", "Charlotte", "Alice", "Matilda", "Will", "Jessica", "Eric", "Chris", "Brian", "Daniel", "Lily", "Bill", "Claribel Dervla", "Daisy Studious", "Tammie Ema", "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski"]


import os
import io
import cv2
import g4f 
import time
import json 
import uuid
import torch
import random
import requests
import numpy as np
from PIL import Image
from TTS.api import TTS
from pprint import pprint 
import google.generativeai as genai
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip







GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyC6N1MVe9WmAFjWMNuXjlaLnYa8eO813t')
SEGMIND_API_KEY = os.environ.get('SEGMIND_API_KEY', 'SG_2d3504ba72dbeac')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', 'ec71cc5fb466bbbeaa935e5a7b001d2')

# Configure Gemini with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Function to fetch image description and script based on a prompt
def fetch_imagedescription_and_script(prompt):
    # Set maximum number of retries
    max_retries = 25

    # Retry loop
    for i in range(max_retries):
        try:
            # Check which language model to use
            if LLM == "Gemini":
                # Use Gemini model
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                output = json.loads(response.text)
            elif LLM == "G4F":
                # Use G4F (GPT-3.5-turbo) model
                response = g4f.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.3,
                    max_tokens=2000,
                    top_p=1,
                    stream=False
                )
                output = json.loads(response)
            else:
                # Raise an error if an invalid LLM choice is provided
                raise ValueError("Invalid LLM choice. Choose either 'Gemini' or 'G4F'.")

            # Print the output for debugging
            pprint(output)

            # Extract image prompts and texts from the output
            image_prompts = [k['image_description'] for k in output]
            texts = [k['text'] for k in output]

            # Return the extracted data
            return image_prompts, texts
        except (json.JSONDecodeError, Exception) as e:
            # Print error message and retry
            print(f"Error: {e}. Retrying...")
            time.sleep(1)  # Wait for 1 second before retrying

    # Raise an exception if all retries fail
    raise Exception(f"Failed after {max_retries} retries")



topic = "Success and Achievement"   # @param {type: "string"}
goal = "Genrate A Engaging Youtube Short Video That will Inspire The Viewer! Catchinh Viewr Attention With YouR Lines Inspiratinal Speech, Each Santance Should Be Motivational and Long enough!"   # @param {type: "string"}

prompt_prefix = """You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:""".format(topic,goal)

sample_output="""
   [
       { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
       { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
       ...
   ]"""

prompt_postinstruction="""By following these instructions, you will create an impactful {} short-form video.
Output:""".format(topic)

prompt = prompt_prefix + sample_output + prompt_postinstruction

image_prompts, texts = fetch_imagedescription_and_script(prompt)
print("image_prompts: ", image_prompts)
print("texts: ", texts)
print (len(texts))


current_uuid = uuid.uuid4()
active_folder = str(current_uuid)
print (active_folder)


def generate_images(prompts, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()
    images_in_current_minute = 0

    for i, prompt in enumerate(prompts):
        final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"

        try:
            if VIDEO_SOURCE == "Hercai":
                url = "https://hercai.onrender.com/v3/text2image"
                response = requests.get(url, params={"prompt": final_prompt, "model": "v3"})
                image_url = response.json()['url']
                image_response = requests.get(image_url)
                image = Image.open(io.BytesIO(image_response.content))
            elif VIDEO_SOURCE == "Segmind":
                if images_in_current_minute >= 5:
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 60:
                        wait_time = 60 - elapsed_time
                        print(f"Waiting {wait_time:.2f} seconds to comply with Segmind rate limit...")
                        time.sleep(wait_time)
                    start_time = time.time()
                    images_in_current_minute = 0

                url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
                data = {
                    "prompt": final_prompt,
                    "samples": 1,
                    "seed": random.randint(1, 1000000),
                    "img_width": 1024,
                    "img_height": 1024,
                }
                headers = {'x-api-key': SEGMIND_API_KEY}
                response = requests.post(url, json=data, headers=headers)
                image = Image.open(io.BytesIO(response.content))
                images_in_current_minute += 1
            else:
                raise ValueError("Invalid VIDEO_SOURCE choice. Choose either 'Hercai' or 'Segmind'.")

            image_filename = os.path.join(output_folder, f"{i + 1}.jpg")
            image.save(image_filename)
            print(f"Image {i + 1}/{len(prompts)} saved as '{image_filename}'")

        except Exception as e:
            print(f"Error generating image {i + 1}: {str(e)}")
            time.sleep(1)

generate_images(image_prompts, active_folder)


def generate_speech(texts, tts_provider="XTTS_V2", language="en", voice="Badr Odhiambo", foldername="output"):
    if tts_provider == "Elevenlabs":
        voice_id = "pNInz6obpgDQGcFmaJgB"

        def generate_speech_with_elevenlabs(text, foldername, filename, voice_id, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
            global api_key_index

            # Cycle through API keys
            api_key = api_keys[api_key_index]
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }

            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost
                }
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 429:  # Handle quota exceeded error
                print("Quota exceeded for current API key. Switching to the next key.")
                api_key_index = (api_key_index + 1) % len(api_keys)
                generate_speech_with_elevenlabs(text, foldername, filename, voice_id, model_id, stability, similarity_boost)  # Retry with the new key
            elif response.status_code != 200:
                print(response.text)
            else:
                file_path = f"{foldername}/{filename}.mp3"
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Text: {text} -> Converted to: {file_path}")

        for i, text in enumerate(texts):
            output_filename = str(i + 1)
            generate_speech_with_elevenlabs(text, foldername, output_filename, voice_id)

    elif tts_provider == "XTTS_V2":
        os.environ["COQUI_TOS_AGREED"] = "1"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts.to(device)

        for i, text in enumerate(texts):
            path = f"{active_folder}/{i + 1}.mp3"
            tts.tts_to_file(text=text, file_path=path, language=language, speaker=voice)
            print(f"Text: {text} -> Converted to: {path}")

    else:
        print("Invalid TTS provider choice. Please choose either 'Elevenlabs' or 'XTTS_V2'.")

# Generate speech with the chosen provider, language, and voice
generate_speech(texts, tts_provider=TTS_PROVIDER, language=LANGUAGE, voice=VOICE)


def create_combined_video_audio(mp3_folder, output_filename, output_resolution=(1080, 1920), fps=24):
    mp3_files = sorted([file for file in os.listdir(mp3_folder) if file.endswith(".mp3")])
    mp3_files = sorted(mp3_files, key=lambda x: int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for mp3_file in mp3_files:
        audio_clip = AudioFileClip(os.path.join(mp3_folder, mp3_file))
        audio_clips.append(audio_clip)

        # Load the corresponding image for each mp3 and set its duration to match the mp3's duration
        img_path = os.path.join(mp3_folder, f"{mp3_file.split('.')[0]}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

        # Resize the original image to 1080x1080
        image_resized = cv2.resize(image, (1080, 1080))

        # Blur the image
        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        # Overlay the original image on the blurred one
        y_offset = (output_resolution[1] - 1080) // 2
        blurred_img[y_offset:y_offset+1080, :] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = final_video.with_audio(final_audio)
    finalpath = mp3_folder+"/"+output_filename

    final_video.write_videofile(finalpath, fps=fps, codec='libx264',audio_codec="aac")


output_filename = "combined_video.mp4"
create_combined_video_audio(active_folder, output_filename)
