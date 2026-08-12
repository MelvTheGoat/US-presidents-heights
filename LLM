---
title: "AI Engineer Interview Q&A — Part 1: LLM Fundamentals"
subtitle: "37 core and architecture questions, answered"
author: "Prepared for Mayungbo Oluwatobi Melvyn"
date: ""
geometry: margin=2.2cm
fontsize: 10.5pt
mainfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
linestretch: 1.2
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyfoot[C]{\thepage}
  - \renewcommand{\headrulewidth}{0pt}
---

# Part 1: LLM Fundamentals

This is the section every source says gets you filtered fastest. Interviewers use it to check whether you understand what's happening inside these models or just know how to call an API. Answers here are written to be *spoken*, not read — say them roughly this length in an interview, then let the interviewer probe deeper.

## Core Concepts

**Q1. How do LLMs work?**
A neural network — almost always a Transformer decoder — is trained to predict the next token in a sequence, given everything before it. At training time it sees enormous amounts of text and adjusts billions of parameters to minimise prediction error. At inference time it repeats that same next-token prediction one token at a time, feeding each generated token back in as new context. Everything else — chat behaviour, instruction-following, reasoning — is a consequence of scale plus post-training (RLHF/DPO/instruction tuning) on top of that one core mechanism.

**Q2. How do transformers work?**
An input sequence is embedded into vectors, positional information is added (since attention itself has no notion of order), and the result passes through a stack of identical blocks. Each block runs self-attention — every token looks at every other token and decides how much to weight it — followed by a position-wise feed-forward network, with residual connections and layer normalisation around both. Stacking many of these blocks lets the model build increasingly abstract representations, and the final layer projects back to a probability distribution over the vocabulary.

**Q3. What is tokenization and how does it affect LLM performance?**
Tokenization splits raw text into the discrete units — subwords, mostly — that the model actually operates on. It affects performance in several concrete ways: rare or unusual words fragment into many tokens, burning more of the context window and more compute for the same information; numbers and code often tokenize inconsistently, which is part of why LLMs are historically weak at arithmetic; and a tokenizer trained on general web text handles domain-specific vocabulary (legal, medical, low-resource languages) poorly, inflating token counts and degrading quality on exactly the content that matters most in a specialised deployment.

**Q4. What is the difference between pre-training and fine-tuning?**
Pre-training is the expensive, general phase — next-token prediction over a huge, broad corpus, teaching the model language, facts, and reasoning patterns in general. Fine-tuning takes that pre-trained model and continues training on a smaller, targeted dataset to specialise it — for a task, a domain, a desired behaviour, or a response style. Pre-training is where most of the capability comes from; fine-tuning shapes and directs that capability.

**Q5. Explain context windows and their limitations.**
The context window is the maximum number of tokens the model can attend to at once — input plus output combined. Limitations: cost and latency scale with context length (attention is roughly quadratic in sequence length in the standard formulation); models exhibit a "lost in the middle" effect where information in the centre of a long context is attended to less reliably than information at the start or end; and a genuinely long document may simply not fit, forcing chunking, summarisation, or retrieval instead of raw context stuffing.

**Q6. What are scaling laws and why do they matter?**
Scaling laws are empirical relationships (most famously from Kaplan et al. and later Chinchilla) describing how model loss improves predictably as you increase parameters, data, and compute together. They matter because they let labs plan training runs rationally instead of guessing — the Chinchilla result specifically showed most large models of the era were under-trained on data relative to their parameter count, and that insight reshaped how frontier models are trained today, favouring more data over just more parameters.

**Q7. What is temperature and top-p sampling? How do they affect outputs?**
Temperature rescales the logits before the softmax — near 0 makes the distribution sharply peaked (near-deterministic, picks the highest-probability token almost every time), higher values flatten it toward more uniform, more random sampling. Top-p (nucleus) sampling instead restricts sampling to the smallest set of tokens whose cumulative probability exceeds p, cutting off the long unlikely tail regardless of how flat or peaked the distribution is. In practice: low temperature and/or low top-p for factual, deterministic tasks; higher values for creative or diverse generation.

**Q8. Explain few-shot learning and chain-of-thought prompting.**
Few-shot learning means giving the model a handful of example input/output pairs directly in the prompt so it infers the task pattern without any gradient update — the model's weights don't change, only its context does. Chain-of-thought prompting asks the model to write out intermediate reasoning steps before the final answer, which measurably improves accuracy on multi-step reasoning and arithmetic tasks, because it gives the model's own generated tokens as additional "working memory" to condition on before committing to an answer.

**Q9. What is KV cache? How does it help in LLM inference?**
During autoregressive generation, each new token's attention computation needs the key and value vectors of every previous token. Without caching, you'd recompute those for the entire sequence on every single new token — quadratic waste. The KV cache stores the key/value vectors for all previously processed tokens so each new token only computes its own K/V and attends against the cached ones. This is the single biggest practical lever in LLM serving throughput and is also the main reason inference is memory-bound rather than compute-bound at scale — the cache grows linearly with sequence length and batch size and has to live in GPU memory.

**Q10. Can you describe the difference between GenAI and traditional programming in the context of solving a real-world problem?**
Traditional programming is deterministic and rule-based: you specify exact logic, and the same input always produces the same output through explicit code paths you wrote. GenAI is probabilistic and learned: you specify a goal or examples, the system infers the mapping from data, and outputs are non-deterministic and only statistically reliable. Practically, this changes how you build and test — traditional software gets unit tests with exact assertions; GenAI systems need evaluation frameworks, golden datasets, and statistical thresholds, because "correct" is a distribution, not a fixed answer.

**Q11. How do you ensure the outputs from large language models are consistent and accurate, especially when dealing with complex multi-step workflows?**
Lower or zero temperature for determinism where it matters; structured output formats (JSON schemas, function calling) so downstream steps can validate mechanically rather than parse free text; explicit self-consistency or verification steps, where the model or a second call checks its own work against source material; breaking a complex task into smaller, independently verifiable sub-steps rather than one large opaque generation; and an evaluation harness with a golden set run in CI so regressions in any step are caught before they compound through the pipeline.

**Q12. What's an RAG model? Explain the complete process.**
Retrieval-Augmented Generation grounds a language model's output in an external corpus rather than relying solely on parametric memory. The pipeline: ingest and chunk source documents, embed those chunks into a vector store (often alongside a sparse index like BM25), embed the incoming query the same way, retrieve the top-k most relevant chunks (often reranked with a cross-encoder), assemble those chunks plus the query into a prompt, and generate an answer instructed to use only the retrieved context — ideally with citations back to the source chunks so the answer is verifiable.

**Q13. What are embeddings?**
Dense numerical vectors that represent the semantic meaning of text (or other data) in a continuous space, such that similar meanings land close together geometrically and dissimilar meanings land far apart, typically measured by cosine similarity. Produced by a model trained (via contrastive learning, usually) so that semantically related pairs are pulled together in vector space and unrelated pairs are pushed apart.

**Q14. How does chunking happen?**
Source documents are split into smaller retrievable units, because embedding an entire document as one vector loses too much granularity to be useful for precise retrieval. Common strategies: fixed-size windows with overlap (simple, structure-blind); recursive/semantic chunking that tries to split on paragraph or sentence boundaries; and structure-aware chunking that respects the document's own hierarchy (headings, sections) so a chunk never straddles a meaningful boundary. Chunk size is a real trade-off — too small loses context, too large dilutes the embedding's precision and wastes context window at generation time.

**Q15. What is the difference between discriminative and generative models?**
A discriminative model learns the boundary between classes directly — it models P(label | input) and is good at classification and prediction but cannot produce new data. A generative model learns the underlying data distribution itself — P(input) or P(input, label) — and can therefore sample new, novel examples that resemble the training distribution. GANs, VAEs, and LLMs are generative; logistic regression and most classifiers are discriminative.

**Q16. What is graph RAG? How does it differ from standard RAG?**
Standard RAG retrieves flat, independent chunks based on vector similarity to the query. GraphRAG instead builds a knowledge graph of entities and their relationships from the corpus first, then retrieval can traverse those relationships — multi-hop queries like "which suppliers does Company X depend on that were also affected by the same regulation" become answerable, because the graph structure captures relationships that similarity search alone cannot see. The cost is a much heavier and more complex ingestion pipeline.

**Q17. What is reflection in the context of LLM agents?**
A pattern where an agent evaluates its own output — or the outcome of an action it took — before finalising or proceeding, and revises based on that self-critique. Concretely: generate a draft, generate a critique of that draft (either via a separate prompt or a separate model), then generate a revision informed by the critique. This measurably improves output quality on many tasks at the cost of extra latency and tokens, and it's a core building block of more sophisticated agent loops.

**Q18. Explain KL divergence.**
Kullback-Leibler divergence measures how much one probability distribution diverges from a second, reference distribution — it is not symmetric, so KL(P‖Q) generally differs from KL(Q‖P). In LLM contexts it shows up constantly: in RLHF, a KL penalty term keeps the fine-tuned policy from drifting too far from the original reference model (preventing reward hacking and catastrophic forgetting); in distillation, it's often the loss that pulls a student model's output distribution toward a teacher's.

**Q19. What is the difference between symbolic and connectionist AI?**
Symbolic AI represents knowledge as explicit rules, logic, and symbols manipulated through formal reasoning — think expert systems and classical planning, interpretable but brittle outside their defined rules. Connectionist AI (neural networks, including LLMs) learns statistical patterns from data through distributed representations across many simple units, powerful and flexible but comparatively opaque. Modern "neurosymbolic" approaches try to combine both — LLM tool-use calling out to a symbolic solver or database is a practical instance of this hybrid.

**Q20. Describe the types of text summarization techniques and when you'd use each.**
Extractive summarization selects and stitches together actual sentences or phrases from the source — safer, always factually grounded in the original text, but can read disjointedly. Abstractive summarization generates genuinely new phrasing that paraphrases and condenses the source — more fluent and human-like, but carries real hallucination risk since the model can introduce claims not actually in the source. Use extractive where factual fidelity is non-negotiable (legal, medical); abstractive where readability matters more and you have grounding/verification in place to catch fabrication.

**Q21. How do you do memory management and context management with LLMs?**
Several complementary techniques rather than one solution: sliding-window context that keeps only the most recent turns; summarisation of older conversation history to compress it into a shorter running summary; retrieval-based memory, where past interactions are embedded and stored, then relevant ones are pulled back in only when actually needed rather than kept in every prompt; and structured memory types for agents specifically — working memory (current task state), episodic memory (past interactions), semantic memory (learned facts), and procedural memory (learned skills or successful action sequences) — each with a different retention and retrieval strategy.

\newpage

## Architecture Deep Dives

**Q22. What is the self-attention mechanism? How does it differ from multi-head attention?**
Self-attention lets every token in a sequence compute a weighted combination of every other token's representation, where the weights (attention scores) are derived from learned Query, Key, and Value projections — a token's Query is compared against every other token's Key to produce weights, which then combine the Values. Multi-head attention runs several of these attention operations in parallel, each with its own learned projections, so different heads can specialise in capturing different kinds of relationships (syntactic, positional, semantic) simultaneously, and their outputs are concatenated and projected back down.

**Q23. What is grouped query attention and how does it differ from standard multi-head attention?**
In standard multi-head attention, every attention head has its own full set of Key and Value projections, which is expensive in memory — the KV cache scales with the number of heads. Grouped Query Attention has multiple Query heads share a smaller number of Key/Value heads (a middle ground between full multi-head attention and the more extreme multi-query attention, where all heads share one KV head). This shrinks the KV cache substantially with only a small quality cost, which is why it's now standard in most production-scale LLMs — it's a direct, practical response to inference being memory-bound.

**Q24. What are the differences between BPE, WordPiece, and character-level tokenization? What are the trade-offs?**
Byte-Pair Encoding builds a vocabulary by iteratively merging the most frequent adjacent byte/character pairs, a simple frequency-driven greedy algorithm. WordPiece is similar but merges based on maximising the training data's likelihood rather than raw frequency, giving slightly different, often more linguistically sensible splits. Character-level tokenization uses individual characters as tokens — no vocabulary to learn, trivially handles any input including out-of-vocabulary words and typos, but produces much longer sequences for the same text, which is expensive given quadratic-ish attention costs. BPE and WordPiece both trade some worst-case robustness for dramatically shorter, more efficient sequences on typical text.

**Q25. Explain the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures. When would you use each?**
Encoder-only models (BERT-family) use bidirectional attention — every token can see every other token in both directions — and are suited to understanding tasks like classification, embeddings, and named-entity recognition, not generation. Decoder-only models (GPT-family, most modern LLMs) use causal, left-to-right attention and are suited to open-ended generation. Encoder-decoder models (T5, original Transformer, translation models) use a bidirectional encoder to build a representation of the input and a causal decoder to generate output conditioned on it — well suited to sequence-to-sequence tasks with a clear, distinct input and output, like translation or structured summarisation.

**Q26. Why are decoder-only models dominant even for non-generation tasks?**
Decoder-only architectures scale exceptionally well and, at sufficient scale, can be prompted or lightly fine-tuned to perform understanding tasks nearly as well as purpose-built encoder models — you get one architecture, one training recipe, and one set of scaling laws to invest engineering effort into, rather than maintaining separate encoder and decoder-only model families. The unification of "generation" and "understanding" under one paradigm (frame everything as next-token prediction, including classification via prompting) simplified the whole ecosystem and is why nearly all frontier general-purpose LLMs today are decoder-only.

**Q27. What is positional encoding and why is it needed in Transformers?**
Self-attention itself is permutation-invariant — it has no inherent notion of token order, since it's just weighted sums over a set. Positional encoding injects order information explicitly, either as fixed sinusoidal patterns added to the input embeddings (original Transformer), learned positional embeddings, or, in most modern models, Rotary Position Embeddings (RoPE), which encode relative position by rotating the Query/Key vectors as a function of position — this generalises better to sequence lengths longer than what the model was trained on than absolute positional schemes do.

**Q28. What are the key MMLU, BigBench, and HumanEval benchmarks? What does each measure and what are its limitations?**
MMLU (Massive Multitask Language Understanding) tests broad academic and professional knowledge across 57 subjects via multiple-choice questions — good breadth signal, but multiple-choice format doesn't test generation quality and is increasingly contaminated by having leaked into training data. BigBench is a large, diverse collection of hundreds of tasks probing reasoning, common sense, and model limitations specifically — useful for finding failure modes, but its sheer breadth makes any single score hard to interpret. HumanEval measures functional code generation correctness (does generated code pass unit tests) — a real, executable signal, but narrow to Python and to relatively self-contained function-level problems, not representative of real-world software engineering.

**Q29. What is the difference between RLHF and DPO? When would you prefer one over the other?**
RLHF (Reinforcement Learning from Human Feedback) trains a separate reward model on human preference data, then uses that reward model to fine-tune the policy via reinforcement learning (typically PPO) — powerful but complex, unstable to train, and requires maintaining multiple models simultaneously. DPO (Direct Preference Optimization) reformulates the same preference-learning objective as a single, direct supervised loss on preference pairs, skipping the separate reward model and RL loop entirely — simpler to implement, more stable, cheaper to train, and now the more common default. RLHF's extra complexity can still pay off when you need more nuanced control over the reward signal than a static preference dataset provides.

**Q30. What is Mixture of Experts (MoE)? How does it improve efficiency?**
Instead of every token passing through the same dense feed-forward layer, an MoE layer contains many "expert" sub-networks and a learned router that selects only a small subset (often just one or two) of experts to process each token. This means the model can have a very large total parameter count — and therefore large capacity — while only activating a small fraction of those parameters per forward pass, giving you much of the quality benefit of a much bigger dense model at a fraction of the inference compute cost per token.

**Q31. How do LLMs actually generate text? Explain the autoregressive decoding process.**
Generation proceeds one token at a time. Given the current sequence (prompt plus whatever has been generated so far), the model produces a probability distribution over the entire vocabulary for the next token. A token is sampled from that distribution (deterministically via greedy/argmax, or stochastically via temperature/top-p/top-k sampling), appended to the sequence, and the whole process repeats — each new token's prediction is conditioned on everything generated before it — until a stop token or length limit is reached.

**Q32. What are decoding strategies like beam search, top-k, and top-p? When do you use each?**
Greedy decoding always picks the single highest-probability token — fast and deterministic, but prone to repetitive, low-diversity output and can miss globally better sequences by being locally greedy. Beam search tracks several candidate sequences ("beams") in parallel and keeps the overall highest-probability sequence at the end — better for tasks with one clearly correct answer, like translation, but can produce bland, generic text and is expensive. Top-k sampling restricts sampling to the k highest-probability tokens; top-p (nucleus) sampling restricts to the smallest set covering cumulative probability p — both introduce controlled randomness for more natural, diverse generation, and top-p is generally preferred because it adapts to how peaked or flat the distribution is at each step, unlike a fixed k.

**Q33. What is FlashAttention and how does it work?**
An exact (not approximate) attention algorithm that computes the same mathematical result as standard attention but restructures the computation to minimise reads and writes to slow GPU high-bandwidth memory, by fusing operations and processing attention in blocks that fit in fast on-chip SRAM, using a numerically stable online softmax so it never needs to materialise the full attention matrix at once. The result is significantly faster attention and dramatically lower memory usage for long sequences, with zero approximation error — which is why it's now near-universal in production LLM training and serving.

**Q34. Why is LLM inference memory-bound?**
For typical batch sizes during generation, the GPU spends most of its time waiting on memory bandwidth — reading model weights and the growing KV cache from GPU memory — rather than being limited by raw compute (FLOPs). Each generation step processes very little new compute per token (one token's worth of matrix multiplications) relative to the sheer volume of weights and cached keys/values that have to be read from memory for that step. This is precisely why techniques like KV cache compression, quantization, and Grouped Query Attention — all of which reduce memory traffic — improve serving throughput more than techniques that only reduce FLOPs.

**Q35. How do stop sequences work in LLMs?**
A stop sequence is a specific string (or the model's own end-of-sequence token) that, when generated, immediately halts decoding. They're used to control output length and structure precisely — for example, stopping generation the moment the model emits a closing delimiter like `</answer>`, or a specific string that signals a structured response is complete, preventing the model from continuing to generate unwanted trailing text past the actual answer.

**Q36. What is the context window and what happens when you exceed it? How do you handle long documents?**
Exceeding the context window means the input (plus desired output) simply doesn't fit — you either get a hard error, or older tokens get silently truncated depending on the API, either way with real risk of losing information the model needed. For long documents you typically don't try to fit everything at once: chunk the document and use retrieval to select only the relevant pieces (RAG), or use hierarchical/recursive summarisation (summarise chunks, then summarise the summaries), or, for genuinely long-context use cases, use a model with a larger native context window while remaining aware that effective attention quality still tends to degrade for content buried in the middle of very long contexts.

**Q37. What risks arise from applying a general-purpose tokenizer to specialized domains like legal or medical text?**
Domain-specific vocabulary (drug names, statute citations, legal terms of art) that's rare in the tokenizer's general training corpus gets fragmented into many subword pieces instead of clean single tokens — this inflates token count and cost, and can degrade the model's ability to treat a domain term as one coherent semantic unit rather than an arbitrary sequence of fragments. It can also cause exact-match or citation-precision failures — a specific section number or drug dosage split awkwardly across tokens is more prone to subtle generation errors than a well-tokenized common word. The fix is domain-adapted tokenizer vocabularies or byte-level fallbacks, though most teams instead work around it by keeping such critical strings out of free generation entirely (retrieved and copied verbatim rather than generated).

\newpage

# What's next

This is Part 1 of 6. The remaining parts, in the order I'd tackle them:

- **Part 2 — RAG Systems and Agents** (your strongest area given your existing projects)
- **Part 3 — Fine-tuning, Evaluation, and ML Fundamentals**
- **Part 4 — System Design (AI-specific and traditional)**
- **Part 5 — Coding Questions**
- **Part 6 — Behavioral, Project Deep Dives, and Take-Homes**

Say "next part" and I'll build Part 2.
