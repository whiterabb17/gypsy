use mem_core::ToolDefinition;
use jsonschema::JSONSchema;
use crate::error::{GypsyError, GypsyResult};

pub struct CommandParser;

impl CommandParser {
    pub fn parse(input: &str) -> Option<(&str, &str)> {
        let input_trimmed = input.trim();
        let parts: Vec<&str> = input_trimmed.split_whitespace().collect();
        if parts.is_empty() { 
            return None; 
        }

        let command = parts[0];
        let args_start = input_trimmed.find(command).unwrap_or(0) + command.len();
        let args_str = input_trimmed.get(args_start..)
            .map(|s| s.trim())
            .unwrap_or("");

        Some((command, args_str))
    }
}

pub struct ToolArgumentParser;

impl ToolArgumentParser {
    pub fn parse(
        args_str: &str,
        def: &ToolDefinition,
    ) -> GypsyResult<serde_json::Value> {
        if args_str.trim().is_empty() {
            return Ok(serde_json::json!({}));
        }

        // 1. Try JSON with validation against schema
        if let Ok(val) = serde_json::from_str(args_str) {
            if Self::validate_against_schema(&val, def).is_ok() {
                return Ok(val);
            }
        }

        // 2. Try key=value with type coercion
        if let Some(props) = def.parameters.get("properties").and_then(|p| p.as_object()) {
            let mut map = serde_json::Map::new();
            if let Some(pairs) = shlex::split(args_str) {
                for pair in pairs {
                    if let Some((k, v)) = pair.split_once('=') {
                        if let Some(param_schema) = props.get(k) {
                            let typed_val = Self::coerce_type(v, param_schema)?;
                            map.insert(k.to_string(), typed_val);
                        }
                    }
                }
            }
            if !map.is_empty() {
                let val = serde_json::Value::Object(map);
                if Self::validate_against_schema(&val, def).is_ok() {
                    return Ok(val);
                }
            }
            
            // 3. Single param positional
            if props.len() == 1 {
                let key = props.keys().next().unwrap();
                let param_schema = &props[key];
                let typed_val = Self::coerce_type(args_str, param_schema)?;
                let val = serde_json::json!({ key: typed_val });
                if Self::validate_against_schema(&val, def).is_ok() {
                    return Ok(val);
                }
            }
        }

        // 4. Default raw coercion
        Ok(serde_json::json!(args_str))
    }

    fn coerce_type(val_str: &str, schema: &serde_json::Value) -> GypsyResult<serde_json::Value> {
        let type_str = schema.get("type").and_then(|t| t.as_str()).unwrap_or("string");
        match type_str {
            "integer" => val_str.parse::<i64>().map(|v| serde_json::json!(v))
                .map_err(|_| GypsyError::ToolError("Invalid integer".into())),
            "number" => val_str.parse::<f64>().map(|v| serde_json::json!(v))
                .map_err(|_| GypsyError::ToolError("Invalid number".into())),
            "boolean" => {
                match val_str.to_lowercase().as_str() {
                    "true" | "yes" | "1" => Ok(serde_json::json!(true)),
                    "false" | "no" | "0" => Ok(serde_json::json!(false)),
                    _ => Err(GypsyError::ToolError("Invalid boolean".into())),
                }
            }
            _ => Ok(serde_json::json!(val_str)),
        }
    }

    fn validate_against_schema(
        val: &serde_json::Value,
        def: &ToolDefinition,
    ) -> GypsyResult<()> {
        let schema_val = serde_json::to_value(&def.parameters)?;
        match JSONSchema::compile(&schema_val) {
            Ok(schema) => {
                if let Err(errors) = schema.validate(val) {
                    let mut error_messages = Vec::new();
                    for error in errors {
                        error_messages.push(format!("{}", error));
                    }
                    return Err(GypsyError::ToolError(format!("Schema validation failed: {}", error_messages.join(", "))));
                }
                Ok(())
            }
            Err(e) => Err(GypsyError::ToolError(format!("Invalid schema: {}", e)))
        }
    }
}
