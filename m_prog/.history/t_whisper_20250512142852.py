import openai

# 🔑 1. API 키 입력 (반드시 본인 키로 바꿔줘야 함)
openai.api_key = "sk-여기에_본인_API키_입력"

# 🔊 2. 변환할 음성 파일 열기
file_path = "/path/to/file/openai.mp3"
with open(file_path, "rb") as file:
    # 📝 3. Whisper 모델로 변환 요청
    transcription = openai.Audio.transcribe(
        model="whisper-1",
        file=file
    )

# 📄 4. 결과 출력
print("변환된 텍스트:")
print(transcription['text'])
