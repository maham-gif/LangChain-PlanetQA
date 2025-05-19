from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load LLM and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Avoid warning: set pad token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Documents for retrieval
documents = [
    "Python is a programming language created by Guido van Rossum.",
    "Monty Python is a British comedy group famous for their sketch show.",
    "The Python programming language was named after Monty Python, not the snake."
]

# Step 1: Create embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)

# Step 2: Build a FAISS index and add document embeddings
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_embeddings.astype(np.float32))

# Step 3: Embed query and retrieve relevant docs
query = "Who created Python Programming Language?"
query_emb = embedder.encode([query])
D, I = index.search(query_emb.astype(np.float32), k=2)
retrieved_docs = [documents[i] for i in I[0]]

# Step 4: Build prompt
context = " ".join(retrieved_docs)
prompt = f"\nContext: {context}\nQuestion: \n{query}\nAnswer:"

# Step 5: Generate response
encoding = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Step 6: Extract only the first answer
generated = output_text[len(prompt):].strip()

# Stop at the first period or newline (safe truncation)
for sep in ['.', '\n']:
    if sep in generated:
        generated = generated.split(sep)[0] + '.'
        break

print("Prompt:", prompt)
print(generated)
