import os
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gptcache import Cache
from gptcache.embedding import Huggingface
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, CacheBase, VectorBase

# Config
CACHE_DB_FILE = "gptcache.db"
FAISS_INDEX_FILE = "faiss_index.bin"
DATASET_PATH = "/content/drive/MyDrive/cache/realistic_temporal_surges.csv"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_CACHE_SIZE = 1000  # Max number of cached entries for LFU

llama_tokenizer = None
llama_model = None
llm_calls = 0

embedding_model = Huggingface(model='sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text: str):
    return embedding_model.to_embeddings(text)

def get_llm_response(query: str):
    global llm_calls, llama_tokenizer, llama_model
    llm_calls += 1
    print(f"LLM call #{llm_calls} for query: {query[:50]}...")

    if llama_model is None:
        print(f"Loading {LLM_MODEL_NAME} model...")
        llama_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llama_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        llama_model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."},
        {"role": "user", "content": query},
    ]

    input_ids = llama_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llama_model.device)

    with torch.no_grad():
        outputs = llama_model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=llama_tokenizer.eos_token_id,
        )

    response_text = llama_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response_text.strip()

def init_gpt_cache():
    # Remove cache files if present (optional)
    if os.path.exists(CACHE_DB_FILE):
        os.remove(CACHE_DB_FILE)
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)

    cache = Cache()

    data_manager = get_data_manager(
        CacheBase('sqlite', sql_url=f'sqlite:///{CACHE_DB_FILE}'),
        VectorBase('faiss', dimension=384)
    )

    cache.init(
        embedding_func=get_embedding,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        pre_embedding_func=lambda x: x
    )
    return cache

class LFUCacheWrapper:
    def __init__(self, cache: Cache, max_size=1000):
        self.cache = cache
        self.max_size = max_size
        self.freq = {}  # query str -> frequency
        self.key_order = []  # to track insertion order (optional)

    def _evict_if_needed(self):
        # Evict until cache size <= max_size
        # Size based on number of keys in frequency dict
        while len(self.freq) > self.max_size:
            # Find least frequently used key
            lfu_key = min(self.freq, key=self.freq.get)
            print(f"Evicting LFU cache entry for query: {lfu_key[:30]}... freq={self.freq[lfu_key]}")
            # Remove from GPTCache backend and freq dict
            # Note: GPTCache does not provide direct delete API, so skip actual delete in DB here
            # For demonstration, just remove from freq dict and rely on DB growing (could implement cleanup later)
            del self.freq[lfu_key]
            try:
                self.key_order.remove(lfu_key)
            except ValueError:
                pass

    def search(self, embedding):
        return self.cache.data_manager.search(embedding)

    def get_response(self, query, similarity_threshold=0.8):
        embedding = self.cache.embedding_func(query)
        results = self.search(embedding)
        if results:
            best_score, data_id = results[0]
            # Higher score means more similar for SearchDistanceEvaluation (distance)
            # Note: depends on your similarity metric, adapt threshold logic if needed
            if best_score > similarity_threshold:
                # Cache hit
                print(f"Cache hit with similarity score: {best_score}")
                cached_response = self.cache.data_manager.get_scalar_data(results[0])
                # Update frequency
                self.freq[query] = self.freq.get(query, 0) + 1
                if query not in self.key_order:
                    self.key_order.append(query)
                return cached_response
        return None

    def save(self, query, response):
        embedding = self.cache.embedding_func(query)
        self.cache.data_manager.save(query, response, embedding)
        self.freq[query] = self.freq.get(query, 0) + 1
        if query not in self.key_order:
            self.key_order.append(query)
        self._evict_if_needed()

def create_cached_llm_with_lfu(cache: Cache, llm_func, max_cache_size=1000):
    lfu_cache = LFUCacheWrapper(cache, max_cache_size)

    def cached_llm(query: str):
        cached_response = lfu_cache.get_response(query)
        if cached_response is not None:
            return cached_response

        print("Cache miss! Calling LLM...")
        response = llm_func(query)
        lfu_cache.save(query, response)
        return response

    return cached_llm

if __name__ == "__main__":
    print("Initializing GPTCache...")
    gpt_cache_instance = init_gpt_cache()
    print("GPTCache initialized.")

    cached_llm = create_cached_llm_with_lfu(gpt_cache_instance, get_llm_response, MAX_CACHE_SIZE)

    # Load queries
    try:
        df = pd.read_csv(DATASET_PATH)
        queries = df['query'].tolist()[:1000]  # Take first 1000 queries
        print(f"Loaded {len(queries)} queries.")
    except Exception as e:
        print(f"Could not load dataset, using example queries: {e}")
        queries = ["Hello, how are you?", "What is the weather today?"]

    total_requests = 0

    print(f"\nStarting simulation with {LLM_MODEL_NAME} LLM and LFU Cache...")
    for i, query_text in enumerate(queries):
        total_requests += 1
        start_time = time.time()

        response = cached_llm(query_text)

        end_time = time.time()
        # Log progress every 100 queries
        if i > 0 and i % 100 == 0:
            cache_hits = total_requests - llm_calls
            hit_rate = cache_hits / total_requests if total_requests > 0 else 0
            print(f"Processed {i}/{len(queries)} queries. Cache hit rate: {hit_rate:.4f}. LLM calls: {llm_calls}")

    cache_hits = total_requests - llm_calls
    final_hit_rate = cache_hits / total_requests if total_requests > 0 else 0

    print("\n--- Simulation Results ---")
    print(f"Total requests: {total_requests}")
    print(f"Cache hits: {cache_hits}")
    print(f"LLM calls: {llm_calls}")
    print(f"Final cache hit rate: {final_hit_rate:.4f}")
