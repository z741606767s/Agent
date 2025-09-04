import hashlib
from contextlib import asynccontextmanager
import re
import os
import tempfile
import requests  # 正确导入requests库

import redis
from openai import OpenAI
import asyncio
import edge_tts
import replicate
import uvicorn
from dotenv import load_dotenv
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.fx.all import SilentAudioClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse

# =============== 配置你的 Key ===============
load_dotenv()
api_key = os.getenv("OPENAPI_KEY")
base_url = os.getenv("OPENAPI_BASE_URL")
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# 设置目录
AUDIO_DIR = os.path.join(os.getcwd(), "generated_audio")
TEMP_DIR = tempfile.gettempdir()  # 使用系统临时目录
os.makedirs(AUDIO_DIR, exist_ok=True)


# =============== 启动fastapi ===============
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.redis_client = redis.asyncio.Redis.from_url(
            'redis://10.32.120.2:56379/2',
            encoding='utf-8',
            decode_responses=True,
            password='foobared'
        )
        # 测试连接
        await app.state.redis_client.ping()
        print("Redis 连接成功")
    except redis.exceptions.AuthenticationError as e:
        print(f"Redis 认证失败: {e}")
    except Exception as e:
        print(f"Redis 连接错误: {e}")

    yield
    if hasattr(app.state, 'redis_client'):
        await app.state.redis_client.close()


app = FastAPI(lifespan=lifespan)


# =============== chat接口 ===============
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(content={"error": "No prompt provided"}, status_code=400)

    try:
        response = await create_video_from_text(prompt)
        return JSONResponse(content={"response": {"video": response}})
    except Exception as e:
        print(f"生成视频时出错: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =============== 1. 生成结构化剧本（含场景描述） ===============
async def generate_script_with_scenes(prompt):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    key = f"video:{prompt_hash}"

    cached_response = await app.state.redis_client.get(key)
    if cached_response:
        return cached_response

    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system",
                 "content": "你是一个编剧。请将输入扩展为一个30秒内的短剧，包含多个场景。每个场景需有画面描述和对话。请使用以下格式：\n场景1：场景描述\n角色1：对话内容\n场景2：场景描述\n角色2：对话内容\n...\n不要使用Markdown格式，不要使用项目符号(*)，直接使用纯文本。"},
                {"role": "user", "content": prompt}
            ]
        )

        print(f"生成剧本: {response.choices[0].message.content}")

        if response.choices[0].message.content:
            await app.state.redis_client.set(key, response.choices[0].message.content, ex=60 * 60 * 24)

        return response.choices[0].message.content
    except Exception as e:
        print(f"剧本生成失败: {e}")
        return None


# =============== 2. 文生图：用 Stable Diffusion 生成场景图 ===============
async def generate_image(prompt, output_file):
    print(f"正在生成画面: {prompt}")
    try:
        # 更新为有效的Stable Diffusion模型版本
        output = replicate.run(
            "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f365628f9f90697fdbe47dbaeefe9d417197",
            input={"prompt": prompt, "width": 1280, "height": 720}
        )
        image_url = output[0]

        # 下载图片
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"图片下载失败，状态码: {response.status_code}")

        with open(output_file, 'wb') as f:
            f.write(response.content)
            print(f"图片下载成功，保存为: {output_file}")
            return image_url
    except Exception as e:
        print(f"图片生成失败: {e}")
        # 创建一个占位图片
        from PIL import Image, ImageDraw
        try:
            img = Image.new('RGB', (1280, 720), color=(73, 109, 137))
            draw = ImageDraw.Draw(img)
            draw.text((640, 360), f"场景: {prompt[:30]}...", fill='white', anchor='mm')
            img.save(output_file)
            return output_file
        except Exception as pil_err:
            print(f"创建占位图片失败: {pil_err}")
            raise


# =============== 3. 文字转语音 ===============
VOICES = {
    "小明": "zh-CN-XiaoyiNeural",
    "小鸟": "zh-CN-XiaoxiaoNeural"
}


async def text_to_speech(text, voice, output_file):
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"语音生成成功: {output_file}")
    except Exception as e:
        print(f"语音生成失败: {e}")
        # 创建一个空的音频文件作为占位符
        import wave
        try:
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b'')
        except Exception as wave_err:
            print(f"创建占位音频失败: {wave_err}")


# =============== 4. 解析剧本并生成媒体 ===============
def parse_script(script):
    """解析剧本，提取场景和对话（兼容Markdown格式）"""
    scenes = []
    lines = script.strip().split('\n')
    current_scene = None

    for line in lines:
        line = line.strip()
        # 跳过空行和解释性文字
        if not line or line.startswith(("好的，这是一个", "**短剧：", "---")):
            continue

        # 检测场景行（兼容Markdown格式）
        scene_match = re.match(r'[\*\s]*场景(\d+)：(.+?)[\*\s]*$', line, re.UNICODE)
        if scene_match:
            if current_scene:
                scenes.append(current_scene)
            scene_desc = scene_match.group(2).strip()
            current_scene = {"desc": scene_desc, "dialogues": []}
            continue

        # 检测对话行（兼容Markdown格式）
        dialogue_match = re.match(r'[\*\s]*([^：]+)：(.+?)[\*\s]*$', line, re.UNICODE)
        if dialogue_match and current_scene:
            char = dialogue_match.group(1).strip()
            text = dialogue_match.group(2).strip()
            if char in VOICES:
                current_scene["dialogues"].append({"char": char, "text": text})

    # 添加最后一个场景
    if current_scene:
        scenes.append(current_scene)

    return scenes


async def create_video_from_text(input_text):
    # 1. 生成剧本
    script = await generate_script_with_scenes(input_text)
    if not script:
        raise ValueError("无法生成剧本")
    print("📜 生成剧本：\n", script)

    # 2. 解析剧本
    scenes = parse_script(script)
    print(f"解析出的场景: {scenes}")

    if not scenes:
        raise ValueError("无法解析剧本，没有找到有效场景")

    # 3. 为每个场景生成图片
    for i, scene in enumerate(scenes):
        img_file = os.path.join(TEMP_DIR, f"scene_{i}.jpg")
        await generate_image(scene["desc"], img_file)
        scene["image"] = img_file
        # 验证图片文件是否存在
        if not os.path.exists(img_file) or os.path.getsize(img_file) == 0:
            raise FileNotFoundError(f"场景图片生成失败: {img_file}")

    # 4. 生成语音并创建视频片段
    audio_clips = []
    video_clips = []
    current_time = 0

    for i, scene in enumerate(scenes):
        print(f"处理场景{i}: {scene['image']}")

        # 创建图片剪辑，明确指定duration参数
        try:
            img_clip = ImageClip(scene["image"], duration=5)  # 直接在构造函数中设置时长
        except Exception as e:
            print(f"创建图片剪辑失败: {e}")
            # 尝试替代方案
            img_clip = ImageClip(scene["image"])
            img_clip = img_clip.set_duration(5)  # 备选方案

        # 生成对话语音
        scene_audio_clips = []
        for dialogue in scene["dialogues"]:
            audio_file = os.path.join(AUDIO_DIR, f"audio_{i}_{dialogue['char']}.mp3")
            await text_to_speech(dialogue["text"], VOICES[dialogue["char"]], audio_file)

            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                try:
                    audio_clip = AudioFileClip(audio_file)
                    scene_audio_clips.append(audio_clip)

                    # 创建字幕
                    txt_clip = TextClip(
                        f"{dialogue['char']}: {dialogue['text']}",
                        fontsize=40, color='white',
                        font='SimHei',
                        size=(1280, 100),
                        method='caption'
                    ).set_position(('center', 'bottom')).set_duration(audio_clip.duration).set_start(current_time)

                    video_clips.append(txt_clip)
                    current_time += audio_clip.duration
                except Exception as e:
                    print(f"处理音频文件失败: {e}")

        # 处理场景时长
        if not scene_audio_clips:
            scene_duration = 3  # 3秒的静默场景
            current_time += scene_duration
            img_clip = img_clip.set_duration(scene_duration)
        else:
            # 调整图片持续时间匹配对话总时长
            scene_duration = sum([clip.duration for clip in scene_audio_clips])
            img_clip = img_clip.set_duration(scene_duration)

            # 添加音频到总音频列表
            for audio_clip in scene_audio_clips:
                audio_clip = audio_clip.set_start(current_time - scene_duration)
                audio_clips.append(audio_clip)

        # 添加图片到视频
        img_clip = img_clip.set_start(current_time - scene_duration)
        video_clips.append(img_clip)

    # 5. 合成音频
    if not audio_clips:
        final_audio = SilentAudioClip(duration=current_time)
    else:
        final_audio = CompositeAudioClip(audio_clips)

    # 6. 合成视频
    if not video_clips:
        raise ValueError("没有生成任何视频片段")

    final_video = CompositeVideoClip(video_clips, size=(1280, 720))
    final_video = final_video.set_audio(final_audio)

    output_path = os.path.join(AUDIO_DIR, "dynamic_story_video.mp4")
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

    print(f"✅ 视频已生成：{output_path}")
    return output_path


# =============== 运行 ===============
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
