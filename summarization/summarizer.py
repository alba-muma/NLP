import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import gc

class TextSummarizer:
    def __init__(self):
        print("Cargando modelo de resumen BART...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)
        print(f"BART cargado en {self.device}")

    def summarize(self, text, ratio=0.5, min_length=20):
        """
        Summarize the given text using BART model
        Args:
            text: Texto a resumir
            ratio: Proporción del texto original para la longitud máxima (0.0 a 1.0)
            min_length: Longitud mínima del resumen
        """
        # Calcular longitud del texto original en tokens
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        input_length = len(inputs["input_ids"][0])
        
        # Calcular max_length como porcentaje del original
        max_length = max(int(input_length * ratio), min_length)
        
        # Generar el resumen
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Asegurar que el resumen termine en punto
        if not summary.endswith('.'):
            last_period = summary.rfind('.')
            if last_period != -1:
                summary = summary[:last_period + 1]
            else:
                summary = summary.rstrip() + '.'
        
        return summary

    def process_batch(self, texts, batch_size=8, ratio=0.5, min_length=20, show_progress=True):
        """
        Process a list of texts in batches efficiently
        Args:
            texts: Lista de textos a resumir
            batch_size: Tamaño del lote
            ratio: Proporción del texto original para la longitud máxima
            min_length: Longitud mínima del resumen
            show_progress: Mostrar barra de progreso
        """
        summaries = []
        total_texts = len(texts)
        num_batches = (total_texts + batch_size - 1) // batch_size  # Redondeo hacia arriba
        
        print(f"Procesando {total_texts} textos en {num_batches} lotes de {batch_size}")
        iterator = tqdm(range(0, total_texts, batch_size), 
                       desc="Procesando lotes",
                       total=num_batches,
                       unit="lote") if show_progress else range(0, total_texts, batch_size)
        
        for i in iterator:
            # Limpiar memoria GPU al inicio de cada lote
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            batch = texts[i:i + batch_size]
            batch_summaries = []
            
            # Procesar cada texto en el lote con torch.no_grad()
            with torch.no_grad():
                for text in batch:
                    # Tokenizar y generar resumen
                    inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
                    input_length = len(inputs["input_ids"][0])
                    max_length = max(int(input_length * ratio), min_length)
                    
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        num_beams=4,
                        max_length=max_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.5,
                        pad_token_id=self.tokenizer.pad_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    # Asegurar que termine en punto
                    if not summary.endswith('.'):
                        last_period = summary.rfind('.')
                        if last_period != -1:
                            summary = summary[:last_period + 1]
                        else:
                            summary = summary.rstrip() + '.'
                    
                    batch_summaries.append(summary)
                    
                    # Limpiar memoria después de cada texto
                    del inputs, summary_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            summaries.extend(batch_summaries)
            
            # Limpiar memoria al final del lote
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return summaries

# Ejemplo de uso
if __name__ == "__main__":
    summarizer = TextSummarizer()
    
    # Texto de ejemplo
    text = """
    We present a Spitzer based census of the IC 348 nebula and embedded star cluster. Our Spitzer census supplemented by ground based spectra has added 42 class II T-Tauri sources to the cluster membership and identified ~20 class 0/I protostars. The population of IC 348 likely exceeds 400 sources after accounting statistically for unidentified diskless members. Our Spitzer census of IC 348 reveals a population of protostars that is anti-correlated spatially with the T-Tauri members, which comprise the centrally condensed cluster around a B star. The protostars are instead found mostly at the cluster periphery about 1 pc from the B star and spread out along a filamentary ridge. We find that the star formation rate in this protostellar ridge is consistent with that rate which built the exposed cluster while the presence of fifteen cold, starless, millimeter cores intermingled with this protostellar population indicates that the IC 348 nebula has yet to finish forming stars. We show that the IC 348 cluster is of order 3-5 crossing times old, and, as evidenced by its smooth radial profile and confirmed mass segregation, is likely relaxed. While it seems apparent that the current cluster configuration is the result of dynamical evolution and its primordial structure has been erased, our findings support a model where embedded clusters are built up from numerous smaller sub-clusters. Finally, the results of our Spitzer census indicate that the supposition that star formation must progress rapidly in a dark cloud should not preclude these observations that show it can be relatively long lived.
    """
    
    summary = summarizer.summarize(text)
    print("\nResumen generado:")
    print(summary)