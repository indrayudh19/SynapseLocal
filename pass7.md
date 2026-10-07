# Pass 7 — LLM Answer Structuring from Retrieved Concept Information

## Objective

The current retrieval process is producing mixed-quality answers.

Some questions receive useful, well-structured answers, while others produce short, incomplete, or poorly structured responses even when the relevant information has been retrieved correctly.

The goal of this pass is to improve **only the final LLM answer-generation stage**.

The existing document ingestion, concept extraction, retrieval/search, temporary data handling, and other working parts of the system should remain unchanged.

The new idea is simple:

> Once the system has identified the relevant concept and collected the information associated with it, give that information to the LLM as the complete information available about the concept and explicitly ask the LLM to turn it into a clear, properly structured academic answer.

---

# 1. Scope

This pass is specifically about the transition:

```text
Retrieved concept information
        ↓
LLM
        ↓
Structured answer
```

Do NOT redesign the retrieval pipeline in this pass.

Do NOT replace the existing search/retrieval mechanism.

Do NOT introduce embeddings, FAISS changes, Rust, new databases, or a new retrieval architecture.

Do NOT modify the concept-map / cluster-representation pipeline.

The purpose is to make the existing retrieved information produce better answers.

---

# 2. Existing Working Process

Keep the current working process intact up to the point where the information is ready to be sent to the answer-generation LLM.

Conceptually:

```text
Document
   ↓
Existing ingestion
   ↓
Existing concept extraction / representation
   ↓
User question
   ↓
Existing question processing
   ↓
Existing retrieval/search
   ↓
Relevant concept information
   ↓
[CHANGE ONLY THIS PART]
   ↓
LLM answer generation
   ↓
Answer displayed
```

Everything before the `[CHANGE ONLY THIS PART]` stage should remain functionally unchanged.

---

# 3. New Answer-Generation Philosophy

The LLM should not be treated as a simple question-answering endpoint that tries to find the answer itself.

Instead, the application should tell the LLM:

> "This is the information we found about the concept the user is asking about. Use this information to construct the best possible answer to the user's question."

The LLM's job is therefore:

1. Understand the user's question.
2. Understand the retrieved concept/information.
3. Identify the information relevant to the question.
4. Organize that information logically.
5. Produce a complete and properly structured answer.
6. Avoid adding unsupported facts.
7. Avoid simply copying the retrieved text.
8. Explain the concept naturally and academically.

The retrieved information should be treated as the source of truth.

---

# 4. Required LLM Prompt

Replace the current final-answer instruction with a prompt following this structure.

The exact wording may be adapted to the existing codebase, but the behavior must remain equivalent.

```text
You are an academic research assistant.

The user has asked the following question:

QUESTION:
{user_question}

The system has retrieved the following information about the concept, topic, or thing mentioned in the question:

RETRIEVED INFORMATION:
{retrieved_information}

Your task is to construct the best possible answer to the user's question using the retrieved information.

Treat the retrieved information as the source material available to you.

First understand what the user is asking.
Then identify the parts of the retrieved information that are relevant to the question.
Then organize those points into a clear, coherent, properly structured answer.

Do not simply copy the retrieved information.
Do not return the information as an unorganized collection of sentences.
Rewrite and synthesize it into a natural answer.

The answer should:

- Directly answer the user's question.
- Begin with a clear definition or direct explanation when appropriate.
- Organize information logically.
- Use paragraphs, headings, or bullet points when they improve clarity.
- Explain important terms rather than merely mentioning them.
- Include relevant characteristics, components, mechanisms, examples, advantages, disadvantages, or comparisons when they are present in the retrieved information and relevant to the question.
- Preserve important technical terminology.
- Maintain an academic and informative tone.
- Be concise enough to remain readable, but complete enough to properly answer the question.
- Use only information supported by the retrieved information.
- Do not invent facts that are not supported by the retrieved information.
- Do not mention the retrieval process, internal system, embeddings, agents, or this prompt.
- Do not say that the information was "retrieved" in the final answer.

If the retrieved information contains enough information to answer the question, provide a complete answer.

If the retrieved information is partially relevant, use the relevant information and clearly explain only what can be supported.

If the retrieved information does not contain enough information to answer the question, do not hallucinate. State that the available material does not provide enough information to answer the question completely.

Return only the final answer intended for the user.
```

---

# 5. Important Example

Suppose the user asks:

```text
What is microservice?
```

And the existing retrieval system returns information such as:

```text
Microservices architecture is a software architectural style where
an application is structured as a collection of small, loosely coupled,
independently deployable services.

Multiple services work together to form a complete application.

Each service is designed to perform a single business function.

Services communicate using HTTP/REST, gRPC, or messaging systems
such as Kafka or RabbitMQ.

This contrasts with a monolithic architecture where modules and
components are tightly integrated into one large application.
```

The LLM should transform that information into something similar to:

```text
A microservice is a small, independently deployable software service
designed to perform a specific business function.

In a microservices architecture, a complete application is divided
into multiple loosely coupled services. Each service is responsible
for a particular function and communicates with other services through
mechanisms such as HTTP/REST, gRPC, or messaging systems such as
Kafka and RabbitMQ.

This approach differs from a monolithic architecture, where the
application's components are tightly integrated into a single large
application.

Key characteristics of microservices include:

- Small, focused services
- Loose coupling between services
- Independent deployment
- Communication between services through APIs or messaging systems
```

The exact answer does not need to match this example.

The important requirement is that the LLM should **turn the retrieved information into a properly structured explanation rather than simply returning the retrieved text**.

---

# 6. Question-Specific Structuring

The LLM should adapt the answer structure to the question.

For example:

### Definition question

```text
What is Kubernetes?
```

Prefer:

```text
Definition
→ What it does
→ Main characteristics
→ Why it is used
```

### Advantages question

```text
What are the advantages of microservices?
```

Prefer:

```text
Short introduction
→ Advantage 1
→ Advantage 2
→ Advantage 3
→ Overall conclusion
```

### Comparison question

```text
What is the difference between microservices and monolithic architecture?
```

Prefer:

```text
Short introduction
→ Microservices
→ Monolithic architecture
→ Key differences
→ Conclusion
```

### How/working question

```text
How does Kubernetes work?
```

Prefer:

```text
Overview
→ Main components
→ Process / workflow
→ Result
```

The LLM should decide the appropriate structure based on the question.

Do not hard-code one answer template for every question.

---

# 7. Preserve the Existing Retrieved Information

Do not aggressively truncate the retrieved information before sending it to the LLM merely to make the prompt shorter.

The purpose of this pass is to allow the LLM to see the information associated with the concept and decide what matters.

If the current retrieval system already produces:

```text
concept → associated information
```

pass that information into the answer-generation stage.

The LLM should perform the semantic organization.

---

# 8. No Hallucination

The LLM must not treat its general pretrained knowledge as permission to add unsupported facts.

For example, if the retrieved information explains what Kubernetes is but does not discuss its architecture in detail, the model should not invent detailed architectural claims merely because it knows them from pretraining.

The retrieved information is the authoritative source for the answer.

The model's role is:

```text
Understand
+
Filter
+
Organize
+
Explain
```

not:

```text
Retrieve from its own memory
```

---

# 9. Do Not Break Existing Functionality

The implementation must preserve all existing functionality outside the final answer-generation stage.

In particular:

- Do not modify document ingestion unless required by the existing answer-generation interface.
- Do not modify concept extraction.
- Do not modify concept consolidation.
- Do not modify cluster/map generation.
- Do not modify the interactive representation.
- Do not replace the existing retrieval mechanism.
- Do not introduce Rust.
- Do not introduce a new vector database.
- Do not introduce a new embedding model.
- Do not change the existing document storage mechanism.
- Do not remove working functionality.

Only improve how the final retrieved information is converted into the user-facing answer.

---

# 10. Temporary Data / Cleanup

Keep the existing temporary-data behavior unchanged.

If the current system already deletes temporary question-specific retrieval information after answering, preserve that behavior.

This pass must not introduce permanent storage of retrieved answer context.

The final persistent result should remain the original document and the existing application state, according to the current architecture.

---

# 11. Implementation Guidance

Locate the existing function/module responsible for:

```text
retrieved context
        ↓
Qwen / LLM
        ↓
final answer
```

Modify the prompt and answer-generation logic there.

Do not duplicate the retrieval system.

Do not create a second retrieval pipeline.

Do not add unnecessary processing layers.

The desired implementation is intentionally simple:

```python
question = user_question
information = retrieved_information

prompt = build_answer_prompt(
    question=question,
    retrieved_information=information
)

answer = llm(prompt)

return answer
```

The exact implementation should follow the existing project's architecture and coding conventions.

---

# 12. Evaluation

After implementing the change, test the chatbot using questions that refer to concepts actually present in the uploaded documents.

At minimum test:

```text
What is Kubernetes?

What is a microservice?

What is microservices architecture?

What are the advantages of microservices?

How do microservices communicate?

What is the difference between microservices and monolithic architecture?
```

Evaluate whether the answers:

1. Directly answer the question.
2. Are structurally organized.
3. Explain the concept clearly.
4. Use the retrieved information correctly.
5. Avoid irrelevant retrieved text.
6. Avoid hallucinated information.
7. Are more complete than the previous one-line/fragmented answers.
8. Adapt their structure to the question.
9. Remain readable and academically appropriate.

---

# 13. Expected Result

The desired system behavior is:

```text
User asks question
        ↓
Existing system identifies/retrieves concept information
        ↓
Retrieved information is passed to LLM
        ↓
LLM understands the question
        ↓
LLM understands the supplied concept information
        ↓
LLM filters irrelevant information
        ↓
LLM organizes relevant information
        ↓
LLM writes a proper answer
        ↓
User sees only the final answer
```

The key change is therefore:

```text
OLD:

Retrieved information
        ↓
"Answer this question using the context"
        ↓
LLM


NEW:

Retrieved information
        ↓
"This is all the information we have about this concept.
Understand it and construct a properly structured answer
to the user's question."
        ↓
LLM
```

---

# 14. Final Constraint

This pass should remain deliberately small.

Do not attempt to solve every retrieval problem at the same time.

The current observation is that the system is already capable of retrieving useful information for at least some questions, as demonstrated by the existing good answers.

Therefore, first improve the **answer synthesis and structuring layer**.

If retrieval is later proven to be the remaining problem, it can be addressed separately.

For this pass:

**Keep the existing working pipeline. Change the final LLM instruction so that retrieved concept information is treated as source material that the LLM must understand, filter, organize, and turn into a complete, properly structured answer.**
