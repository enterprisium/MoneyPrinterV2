import uuid
import os
import io
from faster_whisper import WhisperModel
import requests
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import cv2
import numpy as np
import ffmpeg
from PIL import Image
import random
import json
import g4f
import gradio as gr

# Define default values
DEFAULT_TOPIC = "Success and Achievement"
DEFAULT_GOAL = "Inspire people to overcome challenges, achieve success, and celebrate their victories"
DEFAULT_VIDEO_DIMENSIONS = "1080x1920"
DEFAULT_FONT_COLOR = "#FFFFFF"
DEFAULT_FONT_SIZE = 80
DEFAULT_FONT_NAME = "Nimbus-Sans-Bold"
DEFAULT_TEXT_POSITION = "center"

# Global variables for API keys
segmind_apikey = ""
elevenlabs_apikey = ""

def fetch_imagedescription_and_script(prompt):
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts. Always respond in English."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.3,
        max_tokens=2000,
        top_p=1,
        stream=False
    )

    try:
        output = json.loads(response)
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        print("Raw response:", response)
        return [], []

    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]

    return image_prompts, texts

def generate_images(prompts, fname):
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
    headers = {'x-api-key': segmind_apikey}

    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    currentseed = random.randint(1, 1000000)
    print("seed ", currentseed)

    for i, prompt in enumerate(prompts):
        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        data = {
            "prompt": final_prompt,
            "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
            "style": "hdr",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 30,
            "guidance_scale": 8,
            "strength": 1,
            "seed": currentseed,
            "img_width": 1024,
            "img_height": 1024,
            "refiner": "yes",
            "base64": False
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))
            image_filename = os.path.join(fname, f"{i + 1}.jpg")
            image.save(image_filename)
            print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
        else:
            print(response.text)
            print(f"Error: Failed to retrieve or save image {i + 1}")

def generate_and_save_audio(text, foldername, filename, voice_id, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_apikey
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

    if response.status_code != 200:
        print(response.text)
    else:
        file_path = f"{foldername}/{filename}.mp3"
        with open(file_path, 'wb') as f:
            f.write(response.content)

def create_combined_video_audio(mp3_folder, output_filename, output_resolution=(1080, 1920), fps=24, font_color="#FFFFFF", font_size=80, font_name="Nimbus-Sans-Bold", text_position="center"):
    mp3_files = sorted([file for file in os.listdir(mp3_folder) if file.endswith(".mp3")], key=lambda x: int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for mp3_file in mp3_files:
        audio_clip = AudioFileClip(os.path.join(mp3_folder, mp3_file))
        audio_clips.append(audio_clip)

        img_path = os.path.join(mp3_folder, f"{mp3_file.split('.')[0]}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image, (output_resolution[0], output_resolution[0]))

        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        y_offset = (output_resolution[1] - output_resolution[0]) // 2
        blurred_img[y_offset:y_offset+output_resolution[0], :] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video = final_video.with_audio(final_audio)
    finalpath = os.path.join(mp3_folder, output_filename)

    final_video.write_videofile(finalpath, fps=fps, codec='libx264', audio_codec="aac")

def extract_audio_from_video(outvideo):
    if outvideo is None:
        raise ValueError("Input video is None")
    audiofilename = outvideo.replace(".mp4", '.mp3')
    input_stream = ffmpeg.input(outvideo)
    audio = input_stream.audio
    output_stream = ffmpeg.output(audio, audiofilename)
    output_stream = ffmpeg.overwrite_output(output_stream)
    ffmpeg.run(output_stream)
    return audiofilename

def generate_text_clip(word, start, end, video, font_color, font_size, font_name, text_position):
    txt_clip = (TextClip(word, fontsize=font_size, color=font_color, font=font_name, stroke_width=3, stroke_color='black')
                .with_position(text_position)
                .with_duration(end - start))
    return txt_clip.with_start(start)

def get_word_level_timestamps(model, audioname):
    segments, info = model.transcribe(audioname, word_timestamps=True)
    segments = list(segments)
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': word.word, 'start': word.start, 'end': word.end})
    return wordlevel_info

model_size = "base"
model = WhisperModel(model_size)

def add_captions_to_video(videofilename, wordlevelcaptions, font_color, font_size, font_name, text_position):
    video = VideoFileClip(videofilename)
    clips = [generate_text_clip(item['word'], item['start'], item['end'], video, font_color, font_size, font_name, text_position) for item in wordlevelcaptions]
    final_video = CompositeVideoClip([video] + clips)
    path, old_filename = os.path.split(videofilename)
    finalvideoname = os.path.join(path, "final.mp4")
    final_video.write_videofile(finalvideoname, codec="libx264", audio_codec="aac")
    return finalvideoname

def create_video_with_params(topic, goal, video_dimensions, font_color, font_size, font_name, text_position):
    prompt_prefix = f"""You are tasked with creating a script for a {topic} video that is about 30 seconds.
Your goal is to {goal}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:"""

    sample_output = """
    [
        { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
        { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
        ...
    ]"""

    prompt_postinstruction = f"""By following these instructions, you will create an impactful {topic} short-form video.
    Output:"""

    prompt = prompt_prefix + sample_output + prompt_postinstruction

    image_prompts, texts = fetch_imagedescription_and_script(prompt)

    current_uuid = uuid.uuid4()
    current_foldername = str(current_uuid)

    generate_images(image_prompts, current_foldername)

    voice_id = "pNInz6obpgDQGcFmaJgB"
    for i, text in enumerate(texts):
        output_filename = str(i + 1)
        generate_and_save_audio(text, current_foldername, output_filename, voice_id)

    output_filename = "combined_video.mp4"
    create_combined_video_audio(current_foldername, output_filename, 
                                output_resolution=tuple(map(int, video_dimensions.split('x'))),
                                font_color=font_color, font_size=font_size, 
                                font_name=font_name, text_position=text_position)
    
    output_video_file = os.path.join(current_foldername, output_filename)
    return output_video_file

def add_captions(inputvideo, font_color, font_size, font_name, text_position):
    if inputvideo is None:
        raise gr.Error("Please generate a video first before adding captions.")
    audiofilename = extract_audio_from_video(inputvideo)
    wordlevelinfo = get_word_level_timestamps(model, audiofilename)
    finalvidpath = add_captions_to_video(inputvideo, wordlevelinfo, font_color, font_size, font_name, text_position)
    return finalvidpath

def reset_values():
    return DEFAULT_TOPIC, DEFAULT_GOAL, DEFAULT_VIDEO_DIMENSIONS, DEFAULT_FONT_COLOR, DEFAULT_FONT_SIZE, DEFAULT_FONT_NAME, DEFAULT_TEXT_POSITION

def save_api_keys(segmind_key, elevenlabs_key):
    global segmind_apikey, elevenlabs_apikey
    segmind_apikey = segmind_key
    elevenlabs_apikey = elevenlabs_key
    return "API keys saved successfully!"

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Generate Short-form Videos for YouTube Shorts or Instagram Reels")
    
    with gr.Tab("Video Generation"):
        with gr.Row():
            topic = gr.Textbox(label="Topic", placeholder="Enter the video topic", value=DEFAULT_TOPIC)
            goal = gr.Textbox(label="Goal", placeholder="Enter the video goal", value=DEFAULT_GOAL)
        
        with gr.Row():
            video_dimensions = gr.Dropdown(["1080x1920", "1920x1080"], label="Video Dimensions", value=DEFAULT_VIDEO_DIMENSIONS)
            font_color = gr.ColorPicker(label="Font Color", value=DEFAULT_FONT_COLOR)
            font_size = gr.Slider(minimum=20, maximum=120, step=1, label="Font Size", value=DEFAULT_FONT_SIZE)
        
        with gr.Row():
            font_name = gr.Dropdown(["Nimbus-Sans-Bold", "Arial", "Helvetica", "Times New Roman"], label="Font Name", value=DEFAULT_FONT_NAME)
            text_position = gr.Dropdown(["center", "top", "bottom"], label="Text Position", value=DEFAULT_TEXT_POSITION)
        
        with gr.Row():
            btn_create_video = gr.Button('Generate Video')
            btn_reset = gr.Button('Reset to Default')
        
        with gr.Row():
            video = gr.Video(label="Generated Video", format='mp4', height=720, width=405)
            btn_add_captions = gr.Button('Add Captions')
            final_video = gr.Video(label="Video with Captions", format='mp4', height=720, width=405)
    
    with gr.Tab("API Keys"):
        segmind_key_input = gr.Textbox(label="Segmind API Key", type="password")
        elevenlabs_key_input = gr.Textbox(label="ElevenLabs API Key", type="password")
        save_keys_btn = gr.Button("Save API Keys")
        api_status = gr.Textbox(label="Status", interactive=False)
    
    btn_create_video.click(
        fn=create_video_with_params,
        inputs=[topic, goal, video_dimensions, font_color, font_size, font_name, text_position],
        outputs=[video]
    )
    
    btn_reset.click(
        fn=reset_values,
        inputs=[],
        outputs=[topic, goal, video_dimensions, font_color, font_size, font_name, text_position]
    )
    
    btn_add_captions.click(
        fn=add_captions,
        inputs=[video, font_color, font_size, font_name, text_position],
        outputs=[final_video]
    )

demo.launch(debug=True, enable_queue=True)
