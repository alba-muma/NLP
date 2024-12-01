import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ruta a los archivos descargados
model_path = "C:/Users/albam/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Asignar eos_token como pad_token
tokenizer.pad_token = tokenizer.eos_token

# Configurar el dispositivo (GPU si está disponible)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Función para generar texto
def generate(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.95, stop_token="."):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-procesar la salida para evitar cortes a la mitad
    if stop_token in generated_text:
        generated_text = generated_text[:generated_text.rfind(stop_token) + 1]
    
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text.strip()

# Ejemplo de uso
prompt_base = """You are an AI assistant designed to count words in a text. 
    Example: " The quick brown fox jumps over the lazy dog." -> 9 words.
    Example2: "My cat is very cute." -> 5 words.
    Please count the words in the following text:"""
prompt = prompt_base + "The quick brown fox jumps over the lazy dog." 
print(generate(prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.95))