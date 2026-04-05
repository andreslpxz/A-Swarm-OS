import os
import asyncio
from groq import AsyncGroq

class GroqClientWrapper:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, system_prompt: str = "You are an AI assistant. Si necesitas mayor eficiencia, puedes importar librerías como numpy o math dentro de la función.") -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
            stop=None,
        )

        return completion.choices[0].message.content
