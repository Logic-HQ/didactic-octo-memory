# Prefill vs. Decode

Imagine you are asking an LLM to complete a sentence or write a paragraph. The process of getting the answer involves two distinct phases: **Reading** and **Writing**.

| Feature | Prefill Phase (The Reading/Understanding) | Decode Phase (The Writing/Generating) |
| :--- | :--- | :--- |
| **Goal** | To **understand the entire input context** provided by the user. | To **generate the output token-by-token**, one after the other. |
| **What it does** | The model processes *all* input tokens simultaneously to build a rich internal representation (contextual understanding). | The model predicts the *next most likely token* based on everything generated so far and the initial context. |
| **Process Style** | **Parallel Processing.** Everything is processed at once. | **Sequential/Auto-Regressive.** One step leads to the next. |
| **Input Size** | The entire prompt (e.g., "Tell me about cats.") | The input *plus* all previously generated tokens. |
| **Analogy** | Reading a long chapter to understand the plot and characters. | Writing the next sentence based on what you just read. |

---

### 1. Prefill Phase (The Initial Pass)

**What is it?**
Prefill is the initial, heavy computational step where the entire input prompt or context is fed into the model *at one time*.

**How it works:**
* **Full Context Ingestion:** The model reads the entire sequence of input tokens.
* **Contextualization:** It uses its attention mechanism to establish deep relationships and dependencies between all the words in the prompt. This process builds a rich, compressed numerical representation (the context vector) that encapsulates the meaning of the entire input.
* **Output:** This phase primarily determines *what* the model is going to respond to.

**Analogy:**
If you give the LLM a 500-word article, the Prefill phase is like the AI **reading and fully absorbing** that entire article to grasp its meaning.

### 2. Decode Phase (The Step-by-Step Generation)

**What is it?**
Decode is the iterative process where the model generates its output one token at a time, sequentially. This is how the actual text is produced.

**How it works:**
* **Autoregressive Prediction:** The model looks at all the tokens it has generated *so far*, combined with the original context (from the Prefill phase), and predicts the single most likely next token.
* **Looping:** Once a token is chosen, it is added to the sequence, and the entire new, longer sequence is fed back into the model to predict the *next* token. This repeats until an end-of-sequence token is generated or a stopping condition is met.
* **Output:** This phase produces the actual flowing text (the answer).

**Analogy:**
After reading the chapter (Prefill), the Decode phase is like the AI **writing the next sentence**, then looking at that new sentence, and writing the following one, and so on, until the story is complete.

---



| If you are... | You are in the... | Focus Is On... |
| :--- | :--- | :--- |
| **Giving the prompt** to start the process. | **PREFILL** (Context Understanding) | Processing the **entire input** efficiently. |
| **Getting the answer** from the model. | **DECODE** (Text Generation) | Predicting the **next token sequentially**. |

### The Technical Takeaway

1.  **Efficiency:** Prefill is computationally intensive because it processes a long sequence simultaneously. Decode, while iterative, relies heavily on the rich context established during the Prefill phase.
2.  **Architecture:** Modern Transformer models handle this split elegantly. The initial layer does the heavy lifting of context embedding (Prefill), and the subsequent layers manage the sequential prediction (Decode).

**In short: Prefill tells the model *what* to think; Decode tells the model *how* to write it.**
