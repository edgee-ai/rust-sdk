use crate::{
    error::{Error, Result},
    models::*,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use serde_json::json;
use std::pin::Pin;

/// Input types accepted by the send method
#[derive(Debug, Clone)]
pub enum Input {
    /// Simple text input (converted to a user message)
    Text(String),
    /// Structured input with messages and tools
    Object(InputObject),
}

impl From<String> for Input {
    fn from(s: String) -> Self {
        Input::Text(s)
    }
}

impl From<&str> for Input {
    fn from(s: &str) -> Self {
        Input::Text(s.to_string())
    }
}

impl From<InputObject> for Input {
    fn from(obj: InputObject) -> Self {
        Input::Object(obj)
    }
}

impl From<Vec<Message>> for Input {
    fn from(messages: Vec<Message>) -> Self {
        Input::Object(InputObject::new(messages))
    }
}

/// Parsed input components
struct ParsedInput {
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<serde_json::Value>,
    tags: Option<Vec<String>>,
    compression_model: Option<String>,
    tool_result_trimming: Option<bool>,
    tool_surface_reduction: Option<bool>,
    output_brevity: Option<bool>,
}

impl ParsedInput {
    /// Headers for the compression toggles the caller set; unset ones keep the API key setting.
    fn compression_headers(&self) -> Vec<(&'static str, &'static str)> {
        [
            (
                "X-Edgee-Compression-Tool-Result-Trimming",
                self.tool_result_trimming,
            ),
            (
                "X-Edgee-Compression-Tool-Surface-Reduction",
                self.tool_surface_reduction,
            ),
            ("X-Edgee-Compression-Brevity", self.output_brevity),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|v| (name, if v { "true" } else { "false" })))
        .collect()
    }
}

/// Main client for interacting with the Edgee AI Gateway
#[derive(Debug, Clone)]
pub struct Edgee {
    config: EdgeeConfig,
    client: Client,
}

impl Edgee {
    /// Create a new Edgee client with the given configuration
    pub fn new(config: EdgeeConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    /// Create a new Edgee client from environment variables
    /// Reads EDGEE_API_KEY and optionally EDGEE_BASE_URL
    pub fn from_env() -> Result<Self> {
        let config = EdgeeConfig::from_env()?;
        Ok(Self::new(config))
    }

    /// Create a new Edgee client with just an API key (uses default base URL)
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self::new(EdgeeConfig::new(api_key))
    }

    /// Send a chat completion request (non-streaming)
    ///
    /// # Arguments
    /// * `model` - The model to use (e.g., "anthropic/claude-haiku-4-5", "mistral-large-latest")
    /// * `input` - The input (can be a string, InputObject, or `Vec<Message>`)
    ///
    /// # Example
    /// ```no_run
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use edgee::Edgee;
    ///
    /// let client = Edgee::from_env()?;
    /// let response = client.send("anthropic/claude-haiku-4-5", "Hello, world!").await?;
    /// println!("{}", response.text().unwrap_or(""));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(
        &self,
        model: impl Into<String>,
        input: impl Into<Input>,
    ) -> Result<SendResponse> {
        let input = input.into();
        let parsed = self.parse_input(input);
        let compression_headers = parsed.compression_headers();

        let mut body = json!({
            "model": model.into(),
            "messages": parsed.messages,
            "stream": false,
        });

        if let Some(tools) = parsed.tools {
            body["tools"] = json!(tools);
        }
        if let Some(tool_choice) = parsed.tool_choice {
            body["tool_choice"] = tool_choice;
        }
        if let Some(tags) = parsed.tags {
            body["tags"] = json!(tags);
        }
        if let Some(compression_model) = &parsed.compression_model {
            body["compression_model"] = json!(compression_model);
        }

        let mut request = self
            .client
            .post(format!("{}/v1/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");
        for (name, value) in compression_headers {
            request = request.header(name, value);
        }
        let response = request.json(&body).send().await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::Api { status, message });
        }

        let send_response: SendResponse = response.json().await?;
        Ok(send_response)
    }

    /// Send a chat completion request with streaming
    ///
    /// Returns a stream of chunks that can be processed as they arrive
    ///
    /// # Arguments
    /// * `model` - The model to use (e.g., "anthropic/claude-haiku-4-5", "mistral-large-latest")
    /// * `input` - The input (can be a string, InputObject, or `Vec<Message>`)
    ///
    /// # Example
    /// ```no_run
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use edgee::Edgee;
    /// use tokio_stream::StreamExt;
    ///
    /// let client = Edgee::from_env()?;
    /// let mut stream = client.stream("anthropic/claude-haiku-4-5", "Tell me a story").await?;
    ///
    /// while let Some(chunk) = stream.next().await {
    ///     if let Ok(chunk) = chunk {
    ///         if let Some(text) = chunk.text() {
    ///             print!("{}", text);
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream(
        &self,
        model: impl Into<String>,
        input: impl Into<Input>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let input = input.into();
        let parsed = self.parse_input(input);
        let compression_headers = parsed.compression_headers();

        let mut body = json!({
            "model": model.into(),
            "messages": parsed.messages,
            "stream": true,
        });

        if let Some(tools) = parsed.tools {
            body["tools"] = json!(tools);
        }
        if let Some(tool_choice) = parsed.tool_choice {
            body["tool_choice"] = tool_choice;
        }
        if let Some(tags) = parsed.tags {
            body["tags"] = json!(tags);
        }
        if let Some(compression_model) = &parsed.compression_model {
            body["compression_model"] = json!(compression_model);
        }

        let mut request = self
            .client
            .post(format!("{}/v1/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json");
        for (name, value) in compression_headers {
            request = request.header(name, value);
        }
        let response = request.json(&body).send().await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::Api { status, message });
        }

        let stream = response.bytes_stream();
        let parsed_stream = Self::parse_sse_stream(stream);

        Ok(Box::pin(parsed_stream))
    }

    /// Parse SSE stream into StreamChunk objects
    fn parse_sse_stream(
        stream: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    ) -> impl Stream<Item = Result<StreamChunk>> + Send {
        let mut buffer = String::new();

        stream
            .map(move |result| {
                let bytes = result.map_err(Error::Http)?;
                let text = String::from_utf8_lossy(&bytes);
                buffer.push_str(&text);

                let mut chunks = Vec::new();
                while let Some(pos) = buffer.find("\n\n") {
                    let chunk = buffer[..pos].to_string();
                    buffer.drain(..pos + 2);

                    if chunk.is_empty() {
                        continue;
                    }

                    // Parse SSE format: "data: {...}"
                    for line in chunk.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                continue;
                            }

                            match serde_json::from_str::<StreamChunk>(data) {
                                Ok(parsed_chunk) => chunks.push(Ok(parsed_chunk)),
                                Err(e) => {
                                    // Skip malformed JSON (similar to Python SDK behavior)
                                    eprintln!("Failed to parse chunk: {}", e);
                                }
                            }
                        }
                    }
                }

                Ok(chunks)
            })
            .flat_map(|result: Result<Vec<Result<StreamChunk>>>| match result {
                Ok(chunks) => futures::stream::iter(chunks).boxed(),
                Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
            })
    }

    /// Parse input into components
    fn parse_input(&self, input: Input) -> ParsedInput {
        match input {
            Input::Text(text) => ParsedInput {
                messages: vec![Message::user(text)],
                tools: None,
                tool_choice: None,
                tags: None,
                compression_model: None,
                tool_result_trimming: None,
                tool_surface_reduction: None,
                output_brevity: None,
            },
            #[allow(deprecated)]
            Input::Object(obj) => ParsedInput {
                messages: obj.messages,
                tools: obj.tools,
                tool_choice: obj.tool_choice,
                tags: obj.tags,
                compression_model: obj.compression_model,
                tool_result_trimming: obj.tool_result_trimming,
                tool_surface_reduction: obj.tool_surface_reduction,
                output_brevity: obj.output_brevity,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIM: &str = "x-edgee-compression-tool-result-trimming";
    const SURFACE: &str = "x-edgee-compression-tool-surface-reduction";
    const BREVITY: &str = "x-edgee-compression-brevity";

    fn client_for(server: &mockito::Server) -> Edgee {
        Edgee::new(EdgeeConfig::new("test-key").with_base_url(server.url()))
    }

    #[test]
    fn test_compression_headers_only_for_set_toggles() {
        let client = Edgee::with_api_key("test-key");
        let input = InputObject::new(vec![Message::user("hi")])
            .with_tool_result_trimming(false)
            .with_output_brevity(true);
        let headers = client.parse_input(input.into()).compression_headers();
        assert_eq!(
            headers,
            vec![
                ("X-Edgee-Compression-Tool-Result-Trimming", "false"),
                ("X-Edgee-Compression-Brevity", "true"),
            ]
        );

        let text = client.parse_input("hi".into()).compression_headers();
        assert!(text.is_empty());
    }

    #[tokio::test]
    async fn test_send_forwards_compression_headers() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header(TRIM, "true")
            .match_header(SURFACE, "false")
            .match_header(BREVITY, mockito::Matcher::Missing)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"x","object":"chat.completion","created":0,"model":"gpt-4",
                    "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
            )
            .create_async()
            .await;

        let input = InputObject::new(vec![Message::user("hi")])
            .with_tool_result_trimming(true)
            .with_tool_surface_reduction(false);
        let response = client_for(&server).send("gpt-4", input).await.unwrap();

        assert_eq!(response.text(), Some("ok"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_stream_forwards_compression_headers() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header(SURFACE, "true")
            .match_header(TRIM, mockito::Matcher::Missing)
            .with_header("content-type", "text/event-stream")
            .with_body("data: [DONE]\n\n")
            .create_async()
            .await;

        let input = InputObject::new(vec![Message::user("hi")]).with_tool_surface_reduction(true);
        let mut stream = client_for(&server).stream("gpt-4", input).await.unwrap();
        while stream.next().await.is_some() {}

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_text_input_sends_no_compression_headers() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header(TRIM, mockito::Matcher::Missing)
            .match_header(SURFACE, mockito::Matcher::Missing)
            .match_header(BREVITY, mockito::Matcher::Missing)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"x","object":"chat.completion","created":0,"model":"gpt-4",
                    "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
            )
            .create_async()
            .await;

        client_for(&server).send("gpt-4", "hi").await.unwrap();
        mock.assert_async().await;
    }

    #[test]
    fn test_input_conversions() {
        let _input: Input = "hello".into();
        let _input: Input = "hello".to_string().into();
        let _input: Input = InputObject::new(vec![Message::user("hello")]).into();
        let _input: Input = vec![Message::user("hello")].into();
    }

    #[test]
    fn test_config_from_env() {
        unsafe {
            std::env::set_var("EDGEE_API_KEY", "test-key");
            std::env::set_var("EDGEE_BASE_URL", "https://test.example.com");
        }

        let config = EdgeeConfig::from_env().unwrap();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://test.example.com");

        unsafe {
            std::env::remove_var("EDGEE_API_KEY");
            std::env::remove_var("EDGEE_BASE_URL");
        }
    }

    #[test]
    fn test_config_builder() {
        let config = EdgeeConfig::new("my-key").with_base_url("https://custom.example.com");

        assert_eq!(config.api_key, "my-key");
        assert_eq!(config.base_url, "https://custom.example.com");
    }

    #[test]
    fn test_message_constructors() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.as_deref(), Some("hello"));

        let msg = Message::system("You are helpful");
        assert_eq!(msg.role, Role::System);

        let msg = Message::developer("You are helpful");
        assert_eq!(msg.role, Role::Developer);

        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);

        let msg = Message::tool("call-123", "result");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call-123"));
    }
}
