use colored::Colorize;
use openai_dive::v1::{
    api::Client,
    models::EmbeddingsEngine,
    resources::embedding::{EmbeddingInput, EmbeddingParameters},
};
use serde::Serialize;
use std::str::FromStr;
use std::{env, fs::File};
use std::{fmt::Display, io::Read};

const THRESHOLD: f64 = 70.0;

#[tokio::main]
async fn main() {
    let input = env::args()
        .nth(1)
        .expect("Please provide a word or sentence to analyze. Use the command `generate` to generate the embeddings for the sentiments.");

    if input == "generate" {
        process_generate_command().await;

        return;
    }

    let embedding = generate_embedding(EmbeddingInputType::String(input.clone())).await;

    let embedding = match embedding {
        EmbeddingOutputType::Single(embedding) => embedding,
        _ => panic!("Expected single embedding."),
    };

    let mut max_similarity = 0.0;

    let mut similiarity_dictonary = Vec::<(Sentiment, f64)>::new();

    let emotions = get_emotions().await;

    for (_index, item) in emotions.iter().enumerate() {
        let dot_product = calculate_dot_product(&embedding, &item.embedding).await;

        similiarity_dictonary.push((item.sentiment.clone(), dot_product));

        if max_similarity < dot_product {
            max_similarity = dot_product;
        }
    }

    println!("Input: {}", input.bright_blue().bold().underline());

    let mut similiarity_dictonary: Vec<(Sentiment, f64)> = similiarity_dictonary
        .iter()
        .map(|(sentiment, dot_product)| (sentiment.clone(), 100.0 * (dot_product / max_similarity)))
        .collect();

    similiarity_dictonary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    similiarity_dictonary.iter().for_each(|(sentiment, similarity)| {
        if similarity < &THRESHOLD {
            println!("{:<12} {}", sentiment.to_string(), format!("{:.2}%", similarity).red());
        } else {
            println!(
                "{:<12} {}",
                sentiment.to_string(),
                format!("{:.2}%", similarity).green()
            );
        }
    });
}

async fn process_generate_command() {
    let sentiments = vec![
        Sentiment::Sadness,
        Sentiment::Happiness,
        Sentiment::Fear,
        Sentiment::Anger,
        Sentiment::Suprise,
        Sentiment::Disgust,
    ];

    let mut items = Vec::<Item>::new();

    let text_sentiments = sentiments.iter().map(|sentiment| sentiment.to_string()).collect();

    let embeddings = generate_embedding(EmbeddingInputType::Array(text_sentiments)).await;

    let embeddings = match embeddings {
        EmbeddingOutputType::Multiple(embeddings) => embeddings,
        _ => panic!("Expected multiple embeddings"),
    };

    for (index, sentiment) in sentiments.iter().enumerate() {
        items.push(Item {
            sentiment: sentiment.clone(),
            embedding: embeddings[index].clone(),
        });
    }

    let file_path = "data/embedded-emotions.json";

    let json = serde_json::to_string(&items).unwrap();

    std::fs::write(file_path, json).unwrap();
}

async fn get_emotions() -> Vec<Item> {
    let file_path = "data/embedded-emotions.json";

    let mut file = File::open(file_path).unwrap();

    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();

    let json: serde_json::Value = serde_json::from_str(&data).unwrap();

    let items: Vec<Item> = json
        .as_array()
        .unwrap()
        .iter()
        .map(|item| Item {
            sentiment: Sentiment::from_str(item["sentiment"].as_str().unwrap()).unwrap(),
            embedding: item["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect(),
        })
        .collect();

    items
}

enum EmbeddingInputType {
    String(String),
    Array(Vec<String>),
}

enum EmbeddingOutputType {
    Single(Vec<f64>),
    Multiple(Vec<Vec<f64>>),
}

async fn generate_embedding(input: EmbeddingInputType) -> EmbeddingOutputType {
    let api_key = env::var("OPENAI_API_KEY").expect("$OPENAI_API_KEY is not set");

    let client = Client::new(api_key);

    let formatted_input = match input {
        EmbeddingInputType::String(input) => EmbeddingInput::String(input),
        EmbeddingInputType::Array(input) => EmbeddingInput::StringArray(input),
    };

    let parameters = EmbeddingParameters {
        model: EmbeddingsEngine::TextEmbedding3Small.to_string(),
        input: formatted_input,
        encoding_format: None,
        dimensions: None,
        user: None,
    };

    let embedding_response = client.embeddings().create(parameters).await.unwrap();

    match embedding_response.data.len() {
        1 => EmbeddingOutputType::Single(embedding_response.data[0].embedding.clone()),
        _ => EmbeddingOutputType::Multiple(
            embedding_response
                .data
                .iter()
                .map(|item| item.embedding.clone())
                .collect(),
        ),
    }
}

async fn calculate_dot_product(embedding1: &Vec<f64>, embedding2: &Vec<f64>) -> f64 {
    let mut dot_product: f64 = 0.0;

    for (a, b) in embedding1.iter().zip(embedding2.iter()) {
        dot_product += a * b;
    }

    dot_product
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
enum Sentiment {
    Sadness,
    Happiness,
    Fear,
    Anger,
    Suprise,
    Disgust,
}

#[derive(Debug, Serialize)]
struct Item {
    sentiment: Sentiment,
    embedding: Vec<f64>,
}

impl Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sentiment::Sadness => write!(f, "😔 Sadness"),
            Sentiment::Happiness => write!(f, "😄 Happiness"),
            Sentiment::Fear => write!(f, "😨 Fear"),
            Sentiment::Anger => write!(f, "😠 Anger"),
            Sentiment::Suprise => write!(f, "😮 Suprise"),
            Sentiment::Disgust => write!(f, "🤮 Disgust"),
        }
    }
}

impl FromStr for Sentiment {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sadness" => Ok(Sentiment::Sadness),
            "happiness" => Ok(Sentiment::Happiness),
            "fear" => Ok(Sentiment::Fear),
            "anger" => Ok(Sentiment::Anger),
            "suprise" => Ok(Sentiment::Suprise),
            "disgust" => Ok(Sentiment::Disgust),
            _ => Err(()),
        }
    }
}
