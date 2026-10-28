use std::collections::{HashMap, HashSet};

/// Простой TF-IDF эмбеддер (чистый Rust, без внешних зависимостей)
struct TfIdfEmbedder {
    vocab: HashMap<String, usize>,
    idf: Vec<f32>,
    dim: usize,
}

impl TfIdfEmbedder {
    fn new(documents: &[&str]) -> Self {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut all_words: HashSet<String> = HashSet::new();

        for doc in documents {
            let words: HashSet<String> = tokenize(doc).into_iter().collect();
            for w in &words {
                *doc_freq.entry(w.clone()).or_insert(0) += 1;
                all_words.insert(w.clone());
            }
        }

        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut idf = Vec::new();
        let n_docs = documents.len() as f32;

        for (i, word) in all_words.into_iter().enumerate() {
            vocab.insert(word.clone(), i);
            let df = *doc_freq.get(&word).unwrap_or(&1) as f32;
            idf.push((n_docs / df).ln() + 1.0);
        }

        let dim = vocab.len();
        Self { vocab, idf, dim }
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0; self.dim];
        let tokens = tokenize(text);
        let len = tokens.len().max(1) as f32;

        let mut tf: HashMap<usize, f32> = HashMap::new();
        for t in tokens {
            if let Some(&idx) = self.vocab.get(&t) {
                *tf.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        for (idx, count) in tf {
            vec[idx] = (count / len) * self.idf[idx];
        }

        // L2-нормализация
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.chars().count() > 1)
        .map(|s| s.to_string())
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

fn main() {
    println!("=== Пример текстовых представлений (эмбеддингов) на чистом Rust ===\n");

    let documents = [
        "Машинное обучение позволяет компьютерам учиться на данных.",
        "Глубокое обучение — это подмножество машинного обучения с нейронными сетями.",
        "Кошки любят спать на мягких подушках.",
        "Собаки — лучшие друзья человека.",
        "Rust — системный язык программирования с безопасностью памяти.",
        "Python популярен в области data science и машинного обучения.",
    ];

    println!("Построение TF-IDF модели на {} документах...", documents.len());
    let embedder = TfIdfEmbedder::new(&documents);
    println!("Размерность словаря (эмбеддинга): {}\n", embedder.dim);

    // Генерируем эмбеддинги
    let embeddings: Vec<Vec<f32>> = documents.iter().map(|d| embedder.embed(d)).collect();

    let preview_len = 8.min(embeddings[0].len());
    println!("Первые {} значений первого эмбеддинга: {:?}", preview_len, &embeddings[0][..preview_len]);

    // Косинусное сходство
    println!("\n=== Косинусное сходство (похожие пары) ===");
    for i in 0..documents.len() {
        for j in (i + 1)..documents.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim > 0.05 {
                println!(
                    "{:.3} | «{}...» ↔ «{}...»",
                    sim,
                    truncate_chars(documents[i], 28),
                    truncate_chars(documents[j], 28)
                );
            }
        }
    }

    // Семантический поиск
    println!("\n=== Семантический поиск ===");
    let query = "Как компьютеры учатся на примерах данных?";
    let query_emb = embedder.embed(query);

    let mut results: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(&query_emb, emb)))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Запрос: «{}»\n", query);
    println!("Топ-3 самых релевантных текста:");
    for (rank, (idx, score)) in results.iter().take(3).enumerate() {
        println!("{}. [{:.4}] {}", rank + 1, score, documents[*idx]);
    }

    println!("\n=== Готово! ===");
    println!("Это простой, но полностью рабочий пример текстовых представлений.");
    println!("Для production используйте fastembed / rust-bert / ONNX-модели.");
}
