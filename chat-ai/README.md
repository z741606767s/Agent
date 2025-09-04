# 项目介绍
这是一个基于OpenAI GPT模型的视频生成项目，项目的主要功能是根据用户输入的文本生成视频。

## 所需工具

```text
pip install openai              # 用于文本生成（GPT）
pip install edge-tts            # 免费文字转语音（微软Edge TTS，支持多角色）
pip install moviepy             # 视频合成
pip install pillow              # 图像处理（生成字幕图）
pip install replicate           # 用于调用 Stable Diffusion 生成画面（也可换 DALL·E）
pip install requests
pip install elevenlabs          # 文字转语音（ElevenLabs，支持多角色）

```
#### v1.0 功能
```text
完整流程总结
步骤	          工具	       说明
1. 剧本生成	OpenAI GPT	输入文字 → 结构化短剧
2. 语音合成	edge-tts	不同角色不同音色
3. 视频合成	moviepy	背景 + 字幕 + 音频
4. 输出	MP4 视频	可直接播放分享
```

#### 升级+本地化方案
```text
功能	实现方式
🎞️ 真实视频片段	用文本搜索视频库（Pexels API）
🎨 更高质量图像	使用 DALL·E 3 或 Midjourney
🗣️ 情感语音	    使用 ElevenLabs（支持喜怒哀乐）
🧠 完全本地化  	用本地 LLM（如 Qwen）+ 本地 TTS + 本地 SD

真实视频片段 → 用 Pexels API 搜索真实场景视频 (优点：真实、自然、电影感强 注意：Pexels 免费版有下载限制，商用需授权)
更高质量图像 → 使用 DALL·E 3 生成画面 (目标：比 Stable Diffusion 更精准、更艺术感的画面。 优点：理解力强、构图好、细节丰富。 费用：约 $0.04/张)
情感语音 → 使用 ElevenLabs 情绪化配音 (优点：情感丰富、口型可对齐（未来做数字人）。  费用：按字符计费，适合小段语音)

==================================
技术栈
功能	本地替代方案
📝 剧本生成	Qwen2.5-7B 或 ChatGLM3-6B（本地大模型）
🎨 图像生成	Stable Diffusion WebUI + LoRA（本地跑图）
🗣️ 语音合成	Fish-Speech 或 CosyVoice（本地 TTS）
🎬 视频合成	MoviePy（已本地）

输入文字
   ↓
[本地 LLM] → 生成剧本（含场景+对话+情感标注）
   ↓
[场景1] → [Pexels API 或 Stable Diffusion] → 真实/生成画面
   ↓
[对话+情感] → [ElevenLabs 或 Fish-Speech] → 情感化语音
   ↓
[MoviePy] → 合成：画面 + 音频 + 字幕 + 转场
   ↓
输出：专业级短视频（mp4）
```

```text
# 安装 Fish-Speech
git clone https://github.com/fishaudio/fish-speech
cd fish-speech && pip install -r requirements.txt
# 安装 Fish-Speech 依赖
pip install -r requirements.txt
# 安装 Fish-Speech 模型
python scripts/download_models.py --model fish_speech_v1
```

