from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
else:
    quantization_config = None

# Cargar el modelo y el tokenizador
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",
                                              quantization_config=quantization_config,
                                              device_map="auto" if device == "cuda" else None)

# Asignar eos_token como pad_token
tokenizer.pad_token = tokenizer.eos_token

def cargar_prompts(prompt_file):
    """Lee el contenido de un archivo de prompt"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return content

def generate_text(prompt, max_length, max_new_tokens=200):
    """Genera texto basado en un prompt"""
    # Limpiar y formatear el prompt
    prompt = prompt.strip().encode('utf-8', errors='ignore').decode('utf-8')

    # Tokenizar con longitud limitada
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=0.5,
        num_beams=3,
        do_sample=True,
        #top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # early_stopping=True,
        # length_penalty=-0.5,
        # repetition_penalty=1.5,
        tokenizer=tokenizer,
        stop_strings= ["<STOP>"]
    )
    
    # Decodificar el texto generado
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text.strip()

# Cargar prompts desde el archivo prompt2_0
if __name__ == "__main__":
    # Leer el prompt
    prompt_base = cargar_prompts('./prompts2/prompt2_1')
    
    # Leer el ejemplo específico
    example_path = f"./queries/query_2"
    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            query = f.read()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de ejemplo {example_path}")
        exit(1)
    except Exception as e:
        print(f"Error al leer el archivo de ejemplo: {str(e)}")
        exit(1)
    
    # Generar el prompt completo
    user_query = f"User: {query}"
    full_prompt = (
        prompt_base + '\n' + 
        user_query + '\n' +
        "Assistant:"
    )
    # Contar tokens del prompt
    tokens = tokenizer.encode(full_prompt)
    print(len(tokens))

    print("\nInvestigador:")
    print(user_query)
    generated = generate_text(max_length= len(tokens), prompt=full_prompt)
    print("\nSistema:")
    start = generated.rfind('Assitant:') + len('Assitant:')
    generated = generated.rsplit("STOP>", 1)[0].strip()
    print(generated)
    #[start:generated.rfind('<STOP')].strip())