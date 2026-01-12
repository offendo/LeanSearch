from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from vllm import LLM
import uvicorn

app = FastAPI()
model = LLM("intfloat/e5-mistral-7b-instruct")

@app.post("/")
async def embed(docs: list[str]) -> list[list[float]]:
    outputs = model.embed(docs, use_tqdm=False)
    embeddings = [output.outputs.embedding for output in outputs]
    return embeddings

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
