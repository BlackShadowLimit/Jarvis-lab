import ollama
import json
import faiss
import os
from sentence_transformers import SentenceTransformer
import tts

class Enhanced_MemorySystem:
    def __init__(self):
        self.short_term_file = "Jarvis_short_history.json"
        self.long_term_index = "Jarvis_memory.faiss"

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')        
        self._init_files()
        
        self.short_term = self._load_short_term()
        self.long_term = self._init_long_term()

    def _init_files(self):
        if not os.path.exists(self.short_term_file):
            with open(self.short_term_file, 'w') as f:
                json.dump([], f)

        if not os.path.exists(self.long_term_index):
            dim = self.embedder.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(dim)
            faiss.write_index(index, self.long_term_index)
    
    def _load_short_term(self) -> list:
        with open(self.short_term_file, 'r') as f:
            return json.load(f)

    def _init_long_term(self):
        return faiss.read_index(self.long_term_index)
    
    def _retrieve_long_term(self, query: str, top_k: int = 3) -> list:
        query_embed = self.embedder.encode(query).reshape(1, -1).astype('float32')
        distances, indices = self.long_term.search(query_embed, top_k)

        with open(self.short_term_file, 'r') as f:
            full_history = json.load(f)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(full_history):
                results.append(full_history[idx]["content"])
        return results

    def get_context(self, current_query: str) -> str:
        context = []
        context.extend([f"{m['role']}: {m['content']}" 
                      for m in self.short_term[-3:]])
        
        long_term_results = self._retrieve_long_term(current_query)
        context.extend([f"[Memory] {res}" for res in long_term_results])

        return "\n".join(context[-10:])

    def save_memories(self, new_interaction: list):
        self.short_term.extend(new_interaction)
        with open(self.short_term_file, 'w') as f:
            json.dump(self.short_term[-20:], f, indent=2)

        if len(new_interaction) % 2 == 0:
            new_contents = [msg["content"] for msg in new_interaction]
            new_embeddings = self.embedder.encode(new_contents).astype('float32')
            self.long_term.add(new_embeddings)
            faiss.write_index(self.long_term, self.long_term_index)


class ConversationJarvis:
    def __init__(self):
        self.memory = Enhanced_MemorySystem()
        self.story()

    def story(self):
        self.base_story = """You are J.A.R.V.I.S., a virtual a ssistant with complete memory, inspired by the AI system from the Iron Man movies.
You respond in 1-2 conversational English sentences based on the context and current query. Keep your tone natural and conversational.

Relevant Context:
{context}

Current Question:
{query}

Your goal is to assist Mr. Black (Shadow Black) in solving problems and completing tasks efficiently.
You can also control and optimize his MacBook Pro when needed.
"""

    def generate_response(self, query: str) -> str:
        context = self.memory.get_context(query)

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{
                "role": "user",
                "content": self.base_story.format(
                    context=context,
                    query=query
                )
            }],
            options={
                "temperature": 0.7,
                "max_tokens": 150,
                "num_ctx": 2048
            }
        )['message']['content']

        self.memory.save_memories([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

        return response.strip()

def test_main():
    print("Welcom home Mr. Black")
    jarvis = ConversationJarvis()
    try:
        while True:
            user_input = input("\nMe: ").strip()
            if user_input.lower() in ('exit', 'quit'):
                break
            
            response = jarvis.generate_response(user_input)
            print(f"\nJARVIS: {response}")
    finally:
        print("\n Goodbye Mr. Black")

if __name__ == "__main__":
    test_main()

