use colored::Colorize;
use openai_dive::v1::{
    api::Client,
    models::EmbeddingsEngine,
    resources::embedding::{EmbeddingInput, EmbeddingParameters},
};
use serde::Serialize;
use std::io::Read;
use std::{env, fs::File};

const THRESHOLD: f64 = 50.0;

#[tokio::main]
async fn main() {
    let input = env::args()
        .nth(1)
        .expect("Please provide a string to embed as an argument.");

    if input == "generate" {
        process_generate_command().await;

        return;
    }

    let embedding = generate_embedding(&input).await;

    let mut max_similarity = 0.0;

    let mut similiarity_dictonary = Vec::<(String, f64)>::new();

    let positive_reviews = get_review_embeddings_by_sentiment(&ReviewSentiment::Positive).await;
    let negative_reviews = get_review_embeddings_by_sentiment(&ReviewSentiment::Negative).await;

    for (index, review) in positive_reviews.iter().enumerate() {
        let dot_product = calculate_dot_product(&embedding, review.embedding.as_ref().unwrap()).await;

        similiarity_dictonary.push((format!("POSITIVE-{:?}", index + 1), dot_product));

        if max_similarity < dot_product {
            max_similarity = dot_product;
        }
    }

    // @todo move this duplicated code to it's own function
    for (index, review) in negative_reviews.iter().enumerate() {
        let dot_product = calculate_dot_product(&embedding, review.embedding.as_ref().unwrap()).await;

        similiarity_dictonary.push((format!("NEGATIVE-{:?}", index + 1), dot_product));

        if max_similarity < dot_product {
            max_similarity = dot_product;
        }
    }

    let similiarity_dictonary: Vec<(String, f64)> = similiarity_dictonary
        .iter()
        .map(|(title, dot_product)| (title.to_string(), 100.0 * (dot_product / max_similarity)))
        .collect();

    for item in similiarity_dictonary {
        if item.1 > THRESHOLD {
            println!("{}: {}", item.0, item.1.to_string().green());
        } else {
            println!("{}: {}", item.0, item.1.to_string().red());
        }
    }

    // @todo dertmine the sentiment based on the similarity
}

async fn process_generate_command() {
    let sentiment = env::args()
        .nth(2)
        .expect("Please provide a sentiment (positive or negative) as the 2nd argument.");

    match sentiment.as_str() {
        "positive" => generate_review_embeddings_by_sentiment(ReviewSentiment::Positive).await,
        "negative" => generate_review_embeddings_by_sentiment(ReviewSentiment::Negative).await,
        _ => panic!("Invalid sentiment provided. Please provide either 'positive' or 'negative'."),
    }
}

async fn generate_review_embeddings_by_sentiment(sentiment: ReviewSentiment) {
    let reviews = get_reviews_by_sentiment(&sentiment).await;

    let mut embeddings = Vec::<Review>::new();

    for review in reviews {
        let embedding = generate_embedding(&review.title).await;

        let review_with_embedding = Review {
            title: review.title,
            content: review.content,
            embedding: Some(embedding),
        };

        embeddings.push(review_with_embedding);
    }

    store_embeddings_to_file(&sentiment, embeddings).await;
}

async fn get_reviews_by_sentiment(sentiment: &ReviewSentiment) -> Vec<Review> {
    let file_path = match sentiment {
        ReviewSentiment::Positive => "data/positive-movie-reviews.json",
        ReviewSentiment::Negative => "data/negative-movie-reviews.json",
    };

    let mut file = File::open(file_path).unwrap();

    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();

    let json: serde_json::Value = serde_json::from_str(&data).unwrap();

    let reviews: Vec<Review> = json
        .as_array()
        .unwrap()
        .iter()
        .map(|review| Review {
            title: review["title"].as_str().unwrap().to_string(),
            content: review["content"].as_str().unwrap().to_string(),
            embedding: None,
        })
        .collect();

    reviews
}

async fn get_review_embeddings_by_sentiment(sentiment: &ReviewSentiment) -> Vec<Review> {
    let file_path = match sentiment {
        ReviewSentiment::Positive => "data/positive-movie-reviews-embeddings.json",
        ReviewSentiment::Negative => "data/negative-movie-reviews-embeddings.json",
    };

    let mut file = File::open(file_path).unwrap();

    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();

    let json: serde_json::Value = serde_json::from_str(&data).unwrap();

    let reviews: Vec<Review> = json
        .as_array()
        .unwrap()
        .iter()
        .map(|review| Review {
            title: review["title"].as_str().unwrap().to_string(),
            content: review["content"].as_str().unwrap().to_string(),
            embedding: Some(
                review["embedding"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .collect(),
            ),
        })
        .collect();

    reviews
}

async fn generate_embedding(input: &str) -> Vec<f64> {
    let api_key = env::var("OPENAI_API_KEY").expect("$OPENAI_API_KEY is not set");

    let client = Client::new(api_key);

    let parameters = EmbeddingParameters {
        model: EmbeddingsEngine::TextEmbedding3Small.to_string(),
        input: EmbeddingInput::String(input.to_string()),
        encoding_format: None,
        dimensions: None,
        user: None,
    };

    println!("Generating the embedding for input: {}", input.black().on_white());

    let embedding_response = client.embeddings().create(parameters).await.unwrap();

    embedding_response.data[0].embedding.clone()
}

async fn store_embeddings_to_file(sentiment: &ReviewSentiment, embeddings: Vec<Review>) {
    let file_path = match sentiment {
        ReviewSentiment::Positive => "data/positive-movie-reviews-embeddings.json",
        ReviewSentiment::Negative => "data/negative-movie-reviews-embeddings.json",
    };

    let json = serde_json::to_string(&embeddings).unwrap();

    std::fs::write(file_path, json).unwrap();
}

async fn calculate_dot_product(embedding1: &Vec<f64>, embedding2: &Vec<f64>) -> f64 {
    let mut dot_product: f64 = 0.0;

    for (a, b) in embedding1.iter().zip(embedding2.iter()) {
        dot_product += a * b;
    }

    dot_product
}

enum ReviewSentiment {
    Positive,
    Negative,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct Review {
    title: String,
    content: String,
    embedding: Option<Vec<f64>>,
}