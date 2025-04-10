# 🔍 Job Description Similarity using C++

This project implements a simple NLP pipeline in **C++** to calculate similarity between job descriptions using:

- **TF-IDF vectorization**
- **Cosine Similarity**

It provides a hands-on demonstration of basic natural language processing and information retrieval techniques without using any external machine learning libraries.

---

## 🧠 Overview 

The program performs the following steps:

1. **Text preprocessing**  
   - Lowercasing  
   - Removing punctuation  

2. **Tokenization**  
   - Splits text into words

3. **Vocabulary building**  
   - Collects unique words across all documents

4. **TF-IDF vectorization**  
   - Calculates the importance of each word in each document

5. **Cosine Similarity**  
   - Measures similarity between each pair of job descriptions

---

## 📝 Example Input

```txt
1. Senior software engineer with experience in backend systems.
2. Looking for a backend developer with Java and Python skills.
3. Marketing manager with experience in digital advertising.
4. Engineer with strong backend and cloud development skills.

📊 Cosine Similarity between job descriptions:
Similarity(doc 0 vs doc 1) = 0.53
Similarity(doc 0 vs doc 2) = 0.07
Similarity(doc 0 vs doc 3) = 0.49
Similarity(doc 1 vs doc 2) = 0.09
Similarity(doc 1 vs doc 3) = 0.64
Similarity(doc 2 vs doc 3) = 0.08
