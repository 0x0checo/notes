1、**What is RAG?**

RAG is an architecture that combines retrieval with generation.
Instead of letting the LLM rely only on its internal parameters, the system retrieves the most relevant chunks from an external document store and injects them into the prompt so the answer becomes grounded and factual.
The goal is to reduce hallucinations and make the model respond based on real data.

RAG 的工作流程正如其名，分为三个步骤：

检索 (Retrieval)： 当你提出一个问题（比如：“公司最新的报销政策是什么？”），系统不会直接把问题丢给大模型，而是先去你的外部知识库（比如公司的文档数据库）中进行搜索，找到与“报销政策”最相关的几个段落。

增强 (Augmentation)： 系统将你原来的问题，加上刚刚检索到的那些“相关段落”，一起打包成一个新的、更丰富的提示词（Prompt）。

Prompt 示例： “用户问：公司报销政策是什么？请根据以下参考资料回答：[检索到的政策文档片段]...”

生成 (Generation)： 大模型接收到这个增强后的提示词，阅读参考资料，然后生成最终的答案。

2.**Chunking**
# Chunking Function: Principles and Strategies

## 1. What is Chunking?

Chunking is the process of splitting long documents into smaller, 
retrievable units ("chunks") that can be embedded, stored in a vector 
database, and provided to an LLM in a RAG pipeline.

A chunk is typically a short segment (e.g., 200–400 tokens) containing 
coherent information. Good chunking improves retrieval recall and 
grounds the LLM’s output more reliably.

---

## 2. What Does a Chunking Function Do?

A chunking function takes raw text and produces a list of structured chunks:

**Input**
- raw text  
- chunking strategy  
- parameters: `max_tokens`, `overlap`  
- optional metadata (page, section, part_id, language)

**Output**
- list of chunks, each containing:
  - text  
  - metadata  



# RAG Chunking Strategies

This document outlines the core chunking techniques used in Retrieval-Augmented Generation (RAG) systems. Choosing the right strategy is a critical trade-off that directly impacts **Retrieval Recall** and **Generation Precision**.

## 1. Fixed-size Chunking
This is the baseline approach, often used as a starting point.

* **Principle:** Split text into chunks of a fixed size $N$ (characters or tokens), disregarding content structure.
* **Mechanism:** Usually paired with **Overlap**.
    * *Example:* Chunk Size = 500, Overlap = 50.
    * Chunk 1: `[0:500]`, Chunk 2: `[450:950]`.
* **Pros & Cons:**
    * ✅ **Pros:** Computationally cheap; easy to implement; requires no NLP models.
    * ❌ **Cons:** **Semantic Discontinuity**. It blindly cuts through sentences, names, or logical groupings, potentially losing context (though overlap mitigates this slightly).

## 2. Sliding Window Chunking
A technique often combined with fixed-size chunking to enhance context window retrieval.

* **Principle:** Instead of simple overlap, this approach uses a sliding window to capture granular context or retrieval-time expansion.
* **Granularity:**
    * *Chunk 1:* Sentence A + Sentence B + Sentence C
    * *Chunk 2:* Sentence B + Sentence C + Sentence D
* **Core Value:** **Eliminates Boundary Effects**. It ensures that no critical information is lost simply because it fell on a "cut" line, as every data point will eventually appear in the center of a window.

## 3. Structure-aware Chunking (Recursive)
Also known as **Recursive Character Chunking**, this is currently the **industry standard** for processing structured documents (PDF, Markdown, HTML).

* **Principle:** Respects the document's native structure (Headers, Paragraphs, Lists, Code Blocks) rather than splitting by arbitrary character counts.
* **Workflow:**
    1.  **Parse:** Identify separators (e.g., Markdown `#`, `##` or HTML `<div>`).
    2.  **Recursive Split:** Attempt to split by the largest logical unit (e.g., Chapter). If the chunk is still too large for the token limit, recurse down to the next level (e.g., Paragraph).
    3.  **Integrity:** Ensures tables and code blocks remain intact.
* **Core Value:** **High Semantic Cohesion**. Content within a chunk is logically related, and metadata (headers) can be preserved for better retrieval.

## 4. Semantic Chunking
An advanced, **SOTA (State of the Art)** technique that prioritizes meaning over formatting.

* **Principle:** Splits text based on shifts in semantic meaning rather than physical delimiters.
* **Algorithm:**
    1.  **Sentence Embeddings:** Generate vector embeddings for individual sentences.
    2.  **Similarity Check:** Calculate Cosine Similarity between adjacent sentences.
    3.  **Threshold Split:** If similarity is high, merge sentences. If similarity drops below a threshold (indicating a topic change), create a split.
* **Core Value:** **High Signal-to-Noise Ratio**. Each chunk represents a distinct, complete semantic thought, which is crucial for answering complex questions.

## 5. Multilingual Chunking
Essential for globalized applications to handle language density differences.

* **The Problem:** "Length" is defined differently across languages.
    * *Tokenizer differences:* English relies on spaces; CJK (Chinese/Japanese/Korean) languages are dense and lack spacing.
    * *The Trap:* A 500-character limit is a paragraph in English but could be a short essay in Chinese. Using character counts leads to massive chunks in CJK, diluting retrieval accuracy.
* **Solution:**
    * Use **Language-specific splitters** (e.g., NLTK, SpaCy).
    * **Token-based Counting:** Standardize length using the LLM's tokenizer (e.g., `tiktoken`) rather than raw character counts to ensure consistent information density.

---

## ⚡️ Summary & Comparison

| Strategy | Core Logic | Best Use Case | Cost |
| :--- | :--- | :--- | :--- |
| **Fixed-size** | Hard split by length | Plain text, MVP / Baseline testing | 🟢 Low |
| **Sliding Window** | High overlap | High recall requirements; preventing boundary loss | 🟡 Medium |
| **Structure-aware** | **Document Syntax** | **Standard RAG** (Markdown/PDF/Code) | 🟡 Medium |
| **Semantic** | **Meaning/Topic** | Advanced RAG; High precision needs | 🔴 High (GPU) |
| **Multilingual** | Token/Language specific | Multi-language support (CJK mixed with En) | 🟡 Medium |

### 💡 Recommendation
* **Start with:** **Structure-aware (Recursive)** chunking. It offers the best balance of performance and cost.
* **Upgrade to:** **Semantic Chunking** only if you have unstructured text with shifting topics and require maximum accuracy.


**Metadata Filter（元数据过滤）** 是向量数据库检索中用来\*\*“精确缩小搜索范围”\*\*的关键技术。

如果说**向量搜索（Vector Search）是在做“模糊匹配”**（找意思相近的），那么**元数据过滤**就是在做\*\*“精确筛选”\*\*（找条件完全符合的）。

两者结合，才能实现 RAG 系统的高效与精准。

-----

### 🛍️ 一个秒懂的类比：网购

想象你在淘宝/亚马逊买鞋子：

1.  **向量搜索（Search）：**
    你在搜索框输入：“适合夏天跑步穿的透气运动鞋”。

      * *系统通过语义理解，找出了所有跟“夏天”、“跑步”、“透气”相关的鞋子。*

2.  **元数据过滤（Filter）：**
    你点击了侧边栏的筛选按钮：**“价格 \< 500元”** 且 **“品牌 = 耐克”**。

      * *系统把刚才找到的鞋子里，不符合这两个硬性条件的统统踢掉。*

在这个例子中：

  * 鞋子的图片/描述 = **非结构化数据**（用来做向量搜索）。
  * 价格、品牌、尺码 = **元数据（Metadata）**（用来做过滤）。

-----

### ⚙️ 在 RAG 中起什么作用？

在 RAG 系统中，单纯依赖向量搜索往往不够，Metadata Filter 解决了三个核心问题：

#### 1\. 提升精确度 (Precision)

向量搜索有时候太“发散”了。

  * **场景：** 用户问“2023年的财务报告怎么样？”
  * **无过滤：** 向量搜索可能会找出来 2022年、2021年甚至 2010年的财报，因为它们在语义上和“财务报告”都很像。
  * **有过滤：** 设置 `filter: { year: 2023 }`，直接把其他年份的文档屏蔽掉，确保大模型看到的只有 2023 年的数据。

#### 2\. 权限控制 (Security / Multi-tenancy)

这是企业级 RAG 最重要的应用场景。

  * **场景：** 公司里有 HR 文档（包含薪资）和 技术文档。
  * **问题：** 实习生搜“薪资结构”，向量搜索会诚实地把 CEO 的薪资文档找出来。
  * **解决：** 在搜索时强制加上 Filter：`filter: { user_level: "intern" }`。这样，实习生永远搜不到经理级别的文档，哪怕语义再匹配也不行。

#### 3\. 提升效率 (Performance)

  * 如果你的数据库有 1000 万条数据。
  * **有过滤：** 先通过 `category = "law"` 过滤掉 900 万条医学数据，只在剩下的 100 万条法律数据里做向量搜索，速度大大提升。

-----

### 💻 代码长什么样？（基于 Weaviate）

回到我们之前的“动物”例子。假设我们给数据加了 `habitat`（栖息地）这个元数据。

用户想找：**“海里危险的动物”**。

```python
response = animals_collection.query.near_text(
    query="Dangerous animals",  # 语义部分：找危险的
    limit=3,
    filters=wvc.query.Filter.by_property("habitat").equal("Ocean") # 过滤部分：必须是住在海里的
)
```

**执行逻辑：**

1.  **Lion (狮子)**：语义很“危险”，但在陆地 $\rightarrow$ **排除** ❌
2.  **Shark (鲨鱼)**：语义“危险”，且在海里 $\rightarrow$ **保留** ✅
3.  **Goldfish (金鱼)**：在海里/水里，但语义“不危险” $\rightarrow$ **排除** ❌

-----

### ⚠️ 一个重要的技术细节：Pre-filtering vs Post-filtering

面试或者架构设计时，这一点非常关键。

  * **Post-filtering（后过滤 - ❌ 不推荐）：**
    先搜出 Top 100 个向量，然后再从中把不符合 metadata 的删掉。

      * *风险：* 如果你搜“苹果”，Top 100 全是“水果”，然后你过滤 `category="tech"`。结果就是**0条结果**。因为符合条件的根本没机会进入 Top 100。

  * **Pre-filtering（前过滤 - ✅ 推荐）：**
    在进行向量搜索**之前**（或算法内部同时进行），先锁定符合 metadata 的范围，在这个范围内找 Top K。

      * *结果：* 哪怕符合条件的只有 5 条，它也能精准地把这 5 条找出来。
      * *注：* 现代向量数据库（Weaviate, Pinecone, Milvus）默认都支持高效的 Pre-filtering。

### 总结

**Metadata Filter** 就是给向量搜索加上的\*\*“硬约束”\*\*。
它确保了 RAG 系统不仅能听懂“人话”（语义），还能遵守“规则”（时间、地点、权限、类别）。
