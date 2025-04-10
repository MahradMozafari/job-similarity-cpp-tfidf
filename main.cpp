#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <set>

using namespace std;

// Helper: lowercase & remove punctuation
string preprocess(const string& text) {
    string cleaned;
    for (char c : text) {
        if (isalpha(c) || isspace(c)) {
            cleaned += tolower(c);
        }
    }
    return cleaned;
}

// Tokenizer
vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string word;
    while (ss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// Build vocabulary
set<string> build_vocabulary(const vector<vector<string>>& docs) {
    set<string> vocab;
    for (auto& doc : docs) {
        for (auto& word : doc) {
            vocab.insert(word);
        }
    }
    return vocab;
}

// Compute term frequency
unordered_map<string, double> compute_tf(const vector<string>& doc, const set<string>& vocab) {
    unordered_map<string, double> tf;
    for (const string& word : doc) {
        tf[word]++;
    }
    for (auto& [word, count] : tf) {
        tf[word] = count / doc.size();  // normalize
    }
    return tf;
}

// Compute inverse document frequency
unordered_map<string, double> compute_idf(const vector<vector<string>>& docs, const set<string>& vocab) {
    unordered_map<string, double> idf;
    int N = docs.size();
    for (const string& word : vocab) {
        int df = 0;
        for (const auto& doc : docs) {
            if (find(doc.begin(), doc.end(), word) != doc.end()) {
                df++;
            }
        }
        idf[word] = log10((double)N / (1 + df));  // smooth
    }
    return idf;
}

// Build TF-IDF vector
vector<double> build_vector(const vector<string>& doc, const set<string>& vocab,
                            const unordered_map<string, double>& idf) {
    unordered_map<string, double> tf = compute_tf(doc, vocab);
    vector<double> vec;
    for (const auto& word : vocab) {
        double tfidf = tf[word] * idf.at(word);
        vec.push_back(tfidf);
    }
    return vec;
}

// Compute cosine similarity
double cosine_similarity(const vector<double>& a, const vector<double>& b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

// Entry point
int main() {
    // Sample data
    vector<string> raw_texts = {
        "Senior software engineer with experience in backend systems.",
        "Looking for a backend developer with Java and Python skills.",
        "Marketing manager with experience in digital advertising.",
        "Engineer with strong backend and cloud development skills."
    };

    vector<vector<string>> tokenized_docs;
    for (auto& text : raw_texts) {
        string cleaned = preprocess(text);
        tokenized_docs.push_back(tokenize(cleaned));
    }

    // Build vocabulary and compute IDF
    set<string> vocab = build_vocabulary(tokenized_docs);
    unordered_map<string, double> idf = compute_idf(tokenized_docs, vocab);

    // Build TF-IDF vectors
    vector<vector<double>> vectors;
    for (const auto& doc : tokenized_docs) {
        vectors.push_back(build_vector(doc, vocab, idf));
    }

    // Compare similarities
    cout << "\n📊 Cosine Similarity between job descriptions:\n";
    for (size_t i = 0; i < vectors.size(); ++i) {
        for (size_t j = i + 1; j < vectors.size(); ++j) {
            double sim = cosine_similarity(vectors[i], vectors[j]);
            cout << "Similarity(doc " << i << " vs doc " << j << ") = " << sim << endl;
        }
    }

    return 0;
}
