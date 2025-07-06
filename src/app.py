import gradio as gr
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Load model and vector store
model = SentenceTransformer('all-MiniLM-L6-v2')
client = PersistentClient(path="../vector_store/")
collection = client.get_or_create_collection(name='complaints')
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""

from transformers import pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

def embed_query(question: str):
    return model.encode([question])[0].tolist()

def retrieve_top_k(question: str, k=5):
    query_embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    documents = results['documents'][0]
    return documents

def rag_answer(question, k=5, max_context_length=1500):
    docs = retrieve_top_k(question, k)
    context = "\n\n".join(docs)
    if len(context) > max_context_length:
        context = context[:max_context_length]
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generator(prompt, max_new_tokens=256, do_sample=True)[0]['generated_text']
    answer = response.split("Answer:")[-1].strip()
    return answer, docs[:2]

def chat_interface(question):
    answer, sources = rag_answer(question)
    sources_display = "\n\n".join([f"Source {i+1}: {s[:500]}..." for i, s in enumerate(sources)])
    return answer, sources_display

with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust RAG Chatbot\nAsk a question about customer complaints.")
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Your Question", placeholder="Type your question here...")
            submit_btn = gr.Button("Ask")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            answer = gr.Textbox(label="AI-Generated Answer", interactive=False)
            sources = gr.Textbox(label="Top Retrieved Sources", interactive=False)
    submit_btn.click(chat_interface, inputs=question, outputs=[answer, sources])
    clear_btn.click(lambda: ("", ""), None, [answer, sources])

if __name__ == "__main__":
    demo.launch()