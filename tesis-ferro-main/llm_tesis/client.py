import json

from dotenv import load_dotenv
from openai import OpenAI
from prefect import task
from prefect.cache_policies import NO_CACHE

from llm_tesis.prompt import build_system_prompt, build_user_prompt

DEFAULT_MODEL = "gpt-4o-mini"


@task(name="crear_cliente_openai", log_prints=True)
def create_client():
    load_dotenv()
    print("Cliente OpenAI inicializado")
    return OpenAI()


@task(name="clasificar_fila_llm", log_prints=True, retries=6, retry_delay_seconds=[10, 20, 40, 60, 90, 120], cache_policy=NO_CACHE)
def classify_row(client: OpenAI, row: dict, model: str = DEFAULT_MODEL, few_shot_text: str = "") -> dict:
    system_prompt = build_system_prompt(few_shot_text)
    user_prompt = build_user_prompt(row)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    result = json.loads(content)
    parsed = {
        "default_probability": float(result["default_probability"]),
        "prediction": int(result["prediction"]),
        "reasoning": result.get("reasoning", ""),
    }
    print(f"  -> pred={parsed['prediction']}, prob={parsed['default_probability']:.2f} | {parsed['reasoning'][:80]}")
    return parsed
