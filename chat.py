import os
import httpx
from fastapi import HTTPException 

# from huggingface_hub import login

from openai import OpenAI

# --- Setup for External Services (RAG, LLM, etc.) ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# these variables are only used for OmniScience model
ANSWER_OUTPUT_LENGTH = 1024  # Maximum number of tokens to generate for the answer
MAX_LOOPS = 2  # Maximum number of loops to run the LLM

# HuggingFace is currently not being used
# hf_api_keys = os.getenv("HF_API_KEY")
# login(hf_api_keys)

async def query_llm(prompt: str, model: str, max_output_len: int = 1024) -> str:
    """
    Query the LLM inference server with the given prompt and output token budget.
    
    This function queries the inference server at "http://localhost:8800/v2/models/llama/infer"
    and, if the returned output does not include the end-of-text marker ("<|eot_id|>"),
    it issues a follow-up request (continuation) appending the received output to the original prompt.
    
    Args:
        prompt: The full prompt (including any retrieved context) to send.
        max_output_len: The number of tokens to generate for each call.
        
    Returns:
        The full text response from the LLM including all continuation outputs.
    
    Raises:
        HTTPException: If the inference server does not return a valid response.
    """

    if model == "o3-mini":
        final_response = openai_client.chat.completions.create(model="o3-mini", reasoning_effort="high", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                    }
                ]).choices[0].message.content
    elif model == "OmniScience":
        inference_url = "http://localhost:8800/v2/models/llama/infer"
        final_response = ""
        current_prompt = prompt
        count = 0

        while True:
            if count == 0:
                length = max_output_len
            else:
                length = ANSWER_OUTPUT_LENGTH
            payload = {
                "inputs": [
                    {
                        "name": "prompts",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [[current_prompt]]
                    },
                    {
                        "name": "max_output_len",
                        "shape": [1, 1],
                        "datatype": "INT64",
                        "data": [[length]]
                    },
                    {
                        "name": "output_generation_logits",
                        "shape": [1, 1],
                        "datatype": "BOOL",
                        "data": [[False]]
                    }
                ],
                "outputs": [
                    {"name": "outputs"}
                ]
            }

            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(inference_url, json=payload)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code,
                                    detail=f"LLM query failed: {response.text}")

            result = response.json()
            try:
                new_output = result["outputs"][0]["data"][0]
            except (KeyError, IndexError):
                raise HTTPException(status_code=500, detail="LLM query returned no output")

            final_response += new_output

            # Check if the end-of-text marker is present in the full output.
            if "<|eot_id|>" in final_response:
                final_response = final_response.replace("<|eot_id|>", "")
                break

            # if the answer marker isn't present, append one.
            if "<|start_header_id|>answer" not in new_output and count == 0:
                final_response += "\n<|start_header_id|>answer\n"

            # Update the prompt for the continuation call:
            current_prompt = prompt + "\n" + final_response

            count += 1

            if count >= MAX_LOOPS:
                break
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    return final_response

async def extract_molecule_names(text: str) -> list:
    """
    Use OpenAI's 4o-mini model to extract unique molecule names from the given text.
    Returns a list of molecule names.
    """
    prompt = (
        "Extract and list the unique molecule names mentioned in the following text. "
        "Return them as a semicolon-separated list. Only include each molecule once. "
        "Do not include abbreviations if the full molecule name was provided.\n\n"
        f"Text: {text}"
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        molecules = [mol.strip() for mol in result.split(";") if mol.strip()]
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for mol in molecules:
            if mol not in seen:
                unique.append(mol)
                seen.add(mol)
        return unique
    except Exception as e:
        print("Error extracting molecule names:", e)
        return []