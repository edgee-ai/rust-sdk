//! Example: Token compression with Edgee Gateway SDK
//!
//! This example demonstrates how to:
//! 1. Turn tool-result trimming on for a single request using the builder pattern
//! 2. Access compression metrics from the response
//!
//! Tool-result trimming shortens the output of tool calls (here a long `ls -la`
//! listing) before it reaches the model. The per-request toggles
//! (`with_tool_result_trimming`, `with_tool_surface_reduction`, `with_output_brevity`)
//! override the API key settings for this request only; leave one out to keep the
//! key's setting.

use std::collections::HashMap;

use edgee::{
    Edgee, FunctionCall, FunctionDefinition, InputObject, JsonSchema, Message, Role, Tool, ToolCall,
};

/// A long directory listing, the kind of tool output coding agents send back.
fn ls_output() -> String {
    let lines: Vec<String> = (0..200)
        .map(|i| {
            format!(
                "-rw-r--r--  1 user  staff  {} Jan  1 12:00 src/components/module_{i:03}.tsx",
                1000 + i
            )
        })
        .collect();
    format!("total 800\n{}", lines.join("\n"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client from environment variables (EDGEE_API_KEY)
    let client = Edgee::from_env()?;
    let ls_output = ls_output();

    println!("{}", "=".repeat(70));
    println!("Edgee Token Compression Example");
    println!("{}", "=".repeat(70));
    println!();

    println!("Example: Large tool result with tool-result trimming turned on");
    println!("{}", "-".repeat(70));
    println!("Tool output length: {} characters", ls_output.len());
    println!();

    let assistant_call = Message {
        role: Role::Assistant,
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: "call_1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "Bash".to_string(),
                arguments: r#"{"command":"ls -la src/components"}"#.to_string(),
            },
        }]),
        tool_call_id: None,
    };
    let bash_tool = Tool::function(FunctionDefinition {
        name: "Bash".to_string(),
        description: Some("Run a shell command and return its output.".to_string()),
        parameters: JsonSchema {
            schema_type: "object".to_string(),
            properties: Some(HashMap::from([(
                "command".to_string(),
                serde_json::json!({ "type": "string" }),
            )])),
            required: Some(vec!["command".to_string()]),
            description: None,
        },
    });

    let input = InputObject::new(vec![
        Message::user("How many files are in src/components?"),
        assistant_call,
        Message::tool("call_1", ls_output),
    ])
    .with_tools(vec![bash_tool])
    .with_tool_result_trimming(true);

    let response = client.send("anthropic/claude-haiku-4-5", input).await?;

    println!("Response: {}", response.text().unwrap_or(""));
    println!();

    // Display usage information
    if let Some(usage) = &response.usage {
        println!("Token Usage:");
        println!("  Prompt tokens:     {}", usage.prompt_tokens);
        println!("  Completion tokens: {}", usage.completion_tokens);
        println!("  Total tokens:      {}", usage.total_tokens);
        println!();
    }

    // Display compression information
    if let Some(compression) = &response.compression {
        println!("Compression Metrics:");
        println!("  Saved tokens:  {}", compression.saved_tokens);
        println!("  Reduction:     {:.1}%", compression.reduction);
        println!(
            "  Cost savings:  ${:.3}",
            compression.cost_savings as f64 / 1_000_000.0
        );
        println!("  Time:          {} ms", compression.time_ms);
        if compression.reduction > 0.0 {
            let original_tokens =
                (compression.saved_tokens as f64 * 100.0 / compression.reduction) as u32;
            let tokens_after = original_tokens - compression.saved_tokens;
            println!();
            println!("  💡 Without compression, this request would have used");
            println!("     {original_tokens} input tokens.");
            println!("     With compression, only {tokens_after} tokens were processed!");
        }
    } else {
        println!("No compression data available in response.");
        println!("Note: Compression data is only returned when trimming actually shortened");
        println!("      a tool result.");
    }

    println!();
    println!("{}", "=".repeat(70));

    Ok(())
}
