//! The checkpoint's configuration files: `config.json` (the Qwen3.5 backbone and its NVFP4 quantization),
//! `decision_config.json` (temperature, answer codes, attention mode) and `joint_config.json` (the protocol).
//!
//! Only what serving needs is read; everything else in those files is ignored.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The protocol this server speaks, as pplx_decider declares it (protocol.py's PROTOCOL).
pub const PROTOCOL_VERSION: u64 = 3;
pub const PROTOCOL_TEMPLATE: &str = "joint-question-inline-readout-v1";
pub const MARKER: &str = "<|fim_suffix|>";
pub const MAX_OPTIONS: usize = 255;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: PathBuf, source: serde_json::Error },
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

/// `config.json`'s `text_config`: the Qwen3.5 text backbone.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub attn_output_gate: bool,
}

impl TextConfig {
    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.layer_types[layer] == "full_attention"
    }
    /// Rotary dimensions of a full-attention head (partial_rotary_factor x head_dim).
    pub fn rotary_dim(&self) -> usize {
        let factor = self.rope_parameters.partial_rotary_factor.unwrap_or(1.0);
        ((self.head_dim as f64) * factor).round() as usize
    }
    /// Width of the linear-attention qkv projection: q and k (key heads) and v (value heads).
    pub fn linear_qkv_dim(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim + self.linear_num_value_heads * self.linear_value_head_dim
    }
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}

/// `config.json`'s `vision_config`: the Qwen3.5 vision tower.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub hidden_act: String,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default)]
    pub rope_parameters: Option<VisionRope>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionRope {
    pub rope_theta: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub format: String,
    #[serde(default)]
    pub ignore: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

/// `decision_config.json`: the decision head's calibration and codes.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionConfig {
    pub temperature: f64,
    pub codes: Vec<String>,
    pub token_ids: Vec<u32>,
    pub attention_mode: String,
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default)]
    pub max_pixels: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JointConfig {
    pub version: u64,
    pub template: String,
    pub marker: String,
}

/// `processor_config.json`'s image processor settings (the few that matter for text + still images).
#[derive(Debug, Clone, Deserialize)]
pub struct ImageProcessorConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub size: ImageSize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageSize {
    pub shortest_edge: u64,
    pub longest_edge: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct ProcessorConfig {
    image_processor: ImageProcessorConfig,
}

/// Everything serving needs from a checkpoint directory, validated the way pplx_decider validates it.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub path: PathBuf,
    pub model: ModelConfig,
    pub decision: DecisionConfig,
    pub image_processor: ImageProcessorConfig,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read { path: path.to_path_buf(), source })?;
    serde_json::from_str(&text).map_err(|source| ConfigError::Parse { path: path.to_path_buf(), source })
}

impl Checkpoint {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let joint: JointConfig = read_json(&path.join("joint_config.json"))?;
        if joint.version != PROTOCOL_VERSION || joint.template != PROTOCOL_TEMPLATE || joint.marker != MARKER {
            return Err(ConfigError::Invalid(
                "This checkpoint does not declare the joint inline-readout protocol (version 3)".into(),
            ));
        }
        let decision: DecisionConfig = read_json(&path.join("decision_config.json"))?;
        if decision.attention_mode != "noncausal_full_attention" || decision.pooling.as_deref().unwrap_or("last") != "last" {
            return Err(ConfigError::Invalid("Expected a noncausal checkpoint with per-position readouts".into()));
        }
        if !decision.temperature.is_finite() || decision.temperature <= 0.0 {
            return Err(ConfigError::Invalid("Saved temperature must be positive and finite".into()));
        }
        if decision.codes.len() != MAX_OPTIONS || decision.token_ids.len() != MAX_OPTIONS {
            return Err(ConfigError::Invalid(format!("Expected {MAX_OPTIONS} answer codes")));
        }
        if decision.codes.iter().any(|code| code.is_empty() || !code.bytes().all(|b| b.is_ascii_uppercase())) {
            return Err(ConfigError::Invalid("Unexpected answer codes".into()));
        }
        let model: ModelConfig = read_json(&path.join("config.json"))?;
        let text = &model.text_config;
        if text.layer_types.len() != text.num_hidden_layers {
            return Err(ConfigError::Invalid("layer_types does not match num_hidden_layers".into()));
        }
        let processor: ProcessorConfig = read_json(&path.join("processor_config.json"))?;
        Ok(Self { path: path.to_path_buf(), model, decision, image_processor: processor.image_processor })
    }

    /// The served name: `DECISION_MODEL_NAME`, or the checkpoint folder's name without a quantization
    /// suffix (the NVFP4 build of pplx-decider-v1.2 serves as pplx-decider-v1.2).
    pub fn served_name(&self) -> String {
        if let Ok(name) = std::env::var("DECISION_MODEL_NAME")
            && !name.is_empty()
        {
            return name;
        }
        let folder = self.path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let mut name = folder.to_string();
        for suffix in ["-NVFP4", "-nvfp4", "-FP8", "-fp8"] {
            if let Some(stripped) = name.strip_suffix(suffix) {
                name = stripped.to_string();
            }
        }
        if name.starts_with("pplx-decider") { name } else { "pplx-decider-v1.2".to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_name_strips_quantization_suffix() {
        let path = PathBuf::from("/models/machinez/pplx-decider-v1.2-NVFP4");
        let checkpoint = Checkpoint {
            path,
            model: serde_json::from_str(FIXTURE_MODEL).unwrap(),
            decision: serde_json::from_str(FIXTURE_DECISION).unwrap(),
            image_processor: serde_json::from_str(FIXTURE_PROCESSOR).unwrap(),
        };
        unsafe { std::env::remove_var("DECISION_MODEL_NAME") };
        assert_eq!(checkpoint.served_name(), "pplx-decider-v1.2");
    }

    const FIXTURE_MODEL: &str = r#"{"text_config":{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
        "layer_types":["linear_attention","full_attention"],"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":4,"linear_value_head_dim":4,
        "linear_conv_kernel_dim":4,"rms_norm_eps":1e-6,"vocab_size":100,"rope_parameters":{"rope_theta":10000.0},
        "attn_output_gate":true},
        "vision_config":{"depth":1,"hidden_size":8,"intermediate_size":8,"num_heads":1,"in_channels":3,"patch_size":16,
        "temporal_patch_size":2,"spatial_merge_size":2,"out_hidden_size":8,"num_position_embeddings":4,"hidden_act":"gelu_pytorch_tanh"},
        "image_token_id":3,"vision_start_token_id":1,"vision_end_token_id":2}"#;
    const FIXTURE_DECISION: &str =
        r#"{"temperature":1.0,"codes":["A"],"token_ids":[1],"attention_mode":"noncausal_full_attention"}"#;
    const FIXTURE_PROCESSOR: &str = r#"{"patch_size":16,"merge_size":2,"temporal_patch_size":2,"image_mean":[0.5,0.5,0.5],
        "image_std":[0.5,0.5,0.5],"size":{"shortest_edge":65536,"longest_edge":1638400}}"#;
}
