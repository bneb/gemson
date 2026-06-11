use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{Request, StatusCode},
    response::IntoResponse,
    routing::post,
    Router,
};
use http_body_util::BodyExt;
use reqwest::Client;
use std::{env, sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use tower_http::{catch_panic::CatchPanicLayer, timeout::TimeoutLayer, trace::TraceLayer};
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    client: Client,
    semaphore: Arc<Semaphore>,
}

#[tokio::main]
async fn main() {
    // 1. Observability Tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // 2. Keep-Alive tuning & Upstream Timeouts (Deadlock Bleed mitigation)
    let client = Client::builder()
        .pool_idle_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(2)
        .timeout(Duration::from_secs(300)) // Hard upstream connection timeout
        .build()
        .expect("Failed to build reqwest client");

    let state = AppState {
        client,
        semaphore: Arc::new(Semaphore::new(4)), // Max Concurrent Inference streams
    };

    // 3. Web Framework Setup with Safety Layers
    let app = Router::new()
        .route("/v1/extract", post(extract_handler))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // OOM Protection
        .layer(TimeoutLayer::new(Duration::from_secs(300))) // Absolute timeout for the entire request
        .layer(CatchPanicLayer::new()) // Catch panics gracefully to prevent silent drops
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000")
        .await
        .unwrap();
    info!("Gemson Gateway listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn extract_handler(
    State(state): State<AppState>,
    req: Request<Body>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    
    // 4. Authentication Check
    let auth_header = req.headers().get("Authorization").and_then(|h| h.to_str().ok());
    let expected_key = env::var("GEMSON_API_KEY").unwrap_or_else(|_| "dev".to_string());
    
    if auth_header != Some(&format!("Bearer {}", expected_key)) {
        return Err((StatusCode::UNAUTHORIZED, "Invalid API Key".into()));
    }

    // 5. Bounded Queue (Semaphore with Timeout)
    let permit = match tokio::time::timeout(Duration::from_secs(5), state.semaphore.clone().acquire_owned()).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => return Err((StatusCode::INTERNAL_SERVER_ERROR, "Semaphore closed".into())),
        Err(_) => return Err((StatusCode::TOO_MANY_REQUESTS, "Server at capacity. Backpressure applied.".into())),
    };

    // 6. Strict Deserialization (OOM defense)
    let bytes = req.into_body().collect().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?.to_bytes();
    let mut parsed: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(val) => val,
        Err(e) => return Err((StatusCode::BAD_REQUEST, format!("Malformed JSON payload: {}", e))),
    };
    if let Some(obj) = parsed.as_object_mut() {
        obj.insert("stream".to_string(), serde_json::Value::Bool(true));
    }

    // 7. Forward to Local Engine
    let upstream_req = state.client.post("http://127.0.0.1:8080/completion")
        .json(&parsed)
        .build()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let resp = state.client.execute(upstream_req).await.map_err(|e| {
        error!("Upstream error: {}", e);
        (StatusCode::BAD_GATEWAY, e.to_string())
    })?;

    let status = resp.status();
    let stream = resp.bytes_stream();
    use futures::stream::StreamExt;
    let stream_with_permit = stream.map(move |chunk| {
        let _ = &permit; // Capture permit so it lives as long as the stream
        chunk
    });
    let body = Body::from_stream(stream_with_permit);

    Ok((status, body))
}
