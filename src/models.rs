use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the Edgee client
#[derive(Debug, Clone)]
pub struct EdgeeConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for the API (default: <https://api.edgee.ai>)
    pub base_url: String,
}

impl EdgeeConfig {
    /// Create a new configuration with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.edgee.ai".to_string(),
        }
    }

    /// Set a custom base URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Create configuration from environment variables
    /// Reads EDGEE_API_KEY and optionally EDGEE_BASE_URL
    pub fn from_env() -> crate::Result<Self> {
        let api_key = std::env::var("EDGEE_API_KEY").map_err(|_| crate::Error::MissingApiKey)?;

        let base_url =
            std::env::var("EDGEE_BASE_URL").unwrap_or_else(|_| "https://api.edgee.ai".to_string());

        Ok(Self { api_key, base_url })
    }
}

/// Message role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// Function call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool call information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a developer message
    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: Role::Developer,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a tool response message
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// JSON Schema for function parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Function definition for tool calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: JsonSchema,
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

impl Tool {
    /// Create a new function tool
    pub fn function(function: FunctionDefinition) -> Self {
        Self {
            tool_type: "function".to_string(),
            function,
        }
    }
}

/// Tool choice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Auto-select tools
    Auto,
    /// Never use tools
    None,
    /// Use a specific tool
    Specific {
        r#type: String,
        function: HashMap<String, String>,
    },
}

/// Configuration for the compression model.
/// Only relevant for the `agentic` compression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfiguration {
    /// Compression rate (0.0-1.0). Defaults to 0.8 when not specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<f64>,
    /// Semantic preservation threshold (0-100).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_preservation_threshold: Option<i32>,
}

/// Input for the chat completion request
#[derive(Debug, Clone, Serialize)]
pub struct InputObject {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Compression model for this request (agentic, claude, opencode, cursor, or customer).
    /// Only one compression model per request. Each model is a bundle of strategies.
    /// This is a gateway-internal field and is never sent to providers.
    #[serde(default, skip_serializing)]
    pub compression_model: Option<String>,
    /// Configuration for the compression model (rate, semantic preservation threshold).
    /// Only relevant for the `agentic` compression model.
    /// This is a gateway-internal field and is never sent to providers.
    #[serde(default, skip_serializing)]
    pub compression_configuration: Option<CompressionConfiguration>,
}

impl InputObject {
    /// Create a new input with messages
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: None,
            tool_choice: None,
            tags: None,
            compression_model: None,
            compression_configuration: None,
        }
    }

    /// Add tools to the input
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set tool choice
    pub fn with_tool_choice(mut self, tool_choice: serde_json::Value) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set tags for the request
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set the compression model for this request (agentic, claude, opencode, cursor, customer)
    pub fn with_compression_model(mut self, model: impl Into<String>) -> Self {
        self.compression_model = Some(model.into());
        self
    }

    /// Set the compression configuration (only relevant for agentic model)
    pub fn with_compression_configuration(mut self, config: CompressionConfiguration) -> Self {
        self.compression_configuration = Some(config);
        self
    }
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compression {
    pub saved_tokens: u32,
    pub cost_savings: u64, // micro-units (e.g. 27000 = $0.027)
    pub reduction: f64,    // percentage (e.g. 48 = 48%, may be fractional)
    pub time_ms: u32,      // milliseconds
}

/// Choice in a non-streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Response from a non-streaming request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression: Option<Compression>,
}

impl SendResponse {
    /// Get the text content from the first choice
    pub fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
    }

    /// Get the message from the first choice
    pub fn message(&self) -> Option<&Message> {
        self.choices.first().map(|c| &c.message)
    }

    /// Get the finish reason from the first choice
    pub fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
    }

    /// Get tool calls from the first choice
    pub fn tool_calls(&self) -> Option<&Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
    }
}

/// Delta in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Choice in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Chunk in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

impl StreamChunk {
    /// Get the text content from the first choice delta
    pub fn text(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    /// Get the role from the first choice delta
    pub fn role(&self) -> Option<&Role> {
        self.choices.first().and_then(|c| c.delta.role.as_ref())
    }

    /// Get the finish reason from the first choice
    pub fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_response_with_compression() {
        let json = r#"{
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            },
            "compression": {
                "saved_tokens": 42,
                "cost_savings": 27000,
                "reduction": 48,
                "time_ms": 150
            }
        }"#;

        let response: SendResponse = serde_json::from_str(json).unwrap();
        assert!(response.compression.is_some());
        let compression = response.compression.unwrap();
        assert_eq!(compression.saved_tokens, 42);
        assert_eq!(compression.cost_savings, 27000);
        assert!((compression.reduction - 48.0).abs() < 0.01);
        assert_eq!(compression.time_ms, 150);
    }

    #[test]
    fn test_send_response_without_compression() {
        let json = r#"{
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop"
            }]
        }"#;

        let response: SendResponse = serde_json::from_str(json).unwrap();
        assert!(response.compression.is_none());
    }

    #[test]
    fn test_input_object_with_compression_builder() {
        let input = InputObject::new(vec![Message::user("Hello")])
            .with_compression_model("agentic")
            .with_compression_configuration(CompressionConfiguration {
                rate: Some(0.5),
                semantic_preservation_threshold: Some(60),
            });

        assert_eq!(input.compression_model, Some("agentic".to_string()));
        let config = input.compression_configuration.unwrap();
        assert_eq!(config.rate, Some(0.5));
        assert_eq!(config.semantic_preservation_threshold, Some(60));
    }
}
