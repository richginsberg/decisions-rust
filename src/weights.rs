//! The checkpoint's safetensors: the index, memory-mapped shards, and typed views of each tensor.
//!
//! NVFP4 linear layers (compressed-tensors' `nvfp4-pack-quantized`) come as four tensors: `weight_packed`
//! [N, K/2] u8 (two E2M1 codes per byte, low nibble first), `weight_scale` [N, K/16] float8 e4m3,
//! `weight_global_scale` [] f32 and `input_global_scale` [] f32, with value = code x scale / global_scale.
//! Everything else is bf16 (the linear-attention projections, norms, embeddings, the vision tower) or f32
//! (`A_log`, `dt_bias`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("reading {path}: {source}")]
    Read { path: PathBuf, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: PathBuf, source: serde_json::Error },
    #[error("{path}: {source}")]
    SafeTensors { path: PathBuf, source: safetensors::SafeTensorError },
    #[error("missing tensor {0}")]
    Missing(String),
    #[error("tensor {name}: expected {expected}, found {found}")]
    Dtype { name: String, expected: String, found: String },
    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    Shape { name: String, expected: Vec<usize>, found: Vec<usize> },
}

#[derive(serde::Deserialize)]
struct Index {
    weight_map: HashMap<String, String>,
}

/// One memory-mapped shard; `SafeTensors` borrows from the map, so both live here.
struct Shard {
    _map: Mmap,
    tensors: SafeTensors<'static>,
}

/// All of a checkpoint's tensors, by name, across its shards and `readout.safetensors`.
pub struct Weights {
    shards: Vec<Shard>,
    /// tensor name -> shard index
    locations: HashMap<String, usize>,
}

impl Weights {
    pub fn open(checkpoint: &Path) -> Result<Self, WeightsError> {
        let index_path = checkpoint.join("model.safetensors.index.json");
        let index: Index = {
            let text = std::fs::read_to_string(&index_path).map_err(|source| WeightsError::Read { path: index_path.clone(), source })?;
            serde_json::from_str(&text).map_err(|source| WeightsError::Parse { path: index_path.clone(), source })?
        };
        let mut files: Vec<String> = index.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();
        files.push("readout.safetensors".into());
        let mut shards = Vec::with_capacity(files.len());
        let mut locations = HashMap::new();
        for (i, file) in files.iter().enumerate() {
            let path = checkpoint.join(file);
            let handle = std::fs::File::open(&path).map_err(|source| WeightsError::Read { path: path.clone(), source })?;
            // Safety: the files are the checkpoint's, which nothing writes while the server runs.
            let map = unsafe { Mmap::map(&handle) }.map_err(|source| WeightsError::Read { path: path.clone(), source })?;
            let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(map.as_ptr(), map.len()) };
            let tensors = SafeTensors::deserialize(bytes).map_err(|source| WeightsError::SafeTensors { path: path.clone(), source })?;
            for name in tensors.names() {
                locations.insert(name.to_string(), i);
            }
            shards.push(Shard { _map: map, tensors });
        }
        Ok(Self { shards, locations })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.locations.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.locations.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'_>, WeightsError> {
        let shard = *self.locations.get(name).ok_or_else(|| WeightsError::Missing(name.to_string()))?;
        self.shards[shard].tensors.tensor(name).map_err(|_| WeightsError::Missing(name.to_string()))
    }

    /// The raw bytes of a tensor, checked against a dtype and shape (a `None` dimension accepts any size).
    pub fn bytes(&self, name: &str, dtype: Dtype, shape: &[Option<usize>]) -> Result<(&[u8], Vec<usize>), WeightsError> {
        let view = self.get(name)?;
        if view.dtype() != dtype {
            return Err(WeightsError::Dtype { name: name.into(), expected: format!("{dtype:?}"), found: format!("{:?}", view.dtype()) });
        }
        let found = view.shape().to_vec();
        let matches = found.len() == shape.len() && found.iter().zip(shape).all(|(f, s)| s.is_none_or(|s| s == *f));
        if !matches {
            let expected = shape.iter().map(|s| s.unwrap_or(0)).collect();
            return Err(WeightsError::Shape { name: name.into(), expected, found });
        }
        Ok((view.data(), found))
    }

    /// An f32 scalar (`weight_global_scale`, `input_global_scale`); the checkpoint stores these as shape `[1]`,
    /// older exports as shape `[]`, so one element in any rank is accepted.
    pub fn scalar_f32(&self, name: &str) -> Result<f32, WeightsError> {
        let view = self.get(name)?;
        if view.dtype() != Dtype::F32 {
            return Err(WeightsError::Dtype { name: name.into(), expected: "F32".into(), found: format!("{:?}", view.dtype()) });
        }
        if view.shape().iter().product::<usize>() != 1 {
            return Err(WeightsError::Shape { name: name.into(), expected: vec![1], found: view.shape().to_vec() });
        }
        Ok(f32::from_le_bytes(view.data().try_into().expect("4 bytes")))
    }

    /// The tensors of one NVFP4 linear layer.
    pub fn nvfp4(&self, prefix: &str, out_features: usize, in_features: usize) -> Result<Nvfp4Linear<'_>, WeightsError> {
        let (packed, _) = self.bytes(&format!("{prefix}.weight_packed"), Dtype::U8, &[Some(out_features), Some(in_features / 2)])?;
        let (scale, _) = self.bytes(&format!("{prefix}.weight_scale"), Dtype::F8_E4M3, &[Some(out_features), Some(in_features / 16)])?;
        Ok(Nvfp4Linear {
            packed,
            scale,
            weight_global_scale: self.scalar_f32(&format!("{prefix}.weight_global_scale"))?,
            input_global_scale: self.scalar_f32(&format!("{prefix}.input_global_scale"))?,
            out_features,
            in_features,
        })
    }

    pub fn bf16(&self, name: &str, shape: &[Option<usize>]) -> Result<(&[u8], Vec<usize>), WeightsError> {
        self.bytes(name, Dtype::BF16, shape)
    }
}

/// A compressed-tensors NVFP4 linear layer, as stored: value = code x scale / weight_global_scale.
pub struct Nvfp4Linear<'a> {
    /// [N, K/2]: two E2M1 codes per byte, low nibble first.
    pub packed: &'a [u8],
    /// [N, K/16] float8 e4m3 block scales.
    pub scale: &'a [u8],
    pub weight_global_scale: f32,
    pub input_global_scale: f32,
    pub out_features: usize,
    pub in_features: usize,
}

/// The E2M1 magnitudes by code (sign is bit 3).
pub const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode one float8 e4m3fn byte.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = ((byte >> 3) & 0x0f) as i32;
    let mantissa = (byte & 0x07) as f32;
    if exponent == 0 {
        sign * mantissa / 8.0 * 2f32.powi(-6)
    } else if exponent == 15 && mantissa == 7.0 {
        f32::NAN
    } else {
        sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}

impl Nvfp4Linear<'_> {
    /// Dequantize one output row to f32 (the reference for kernel tests; the GPU never does this).
    pub fn row_f32(&self, row: usize) -> Vec<f32> {
        let k = self.in_features;
        let packed = &self.packed[row * k / 2..(row + 1) * k / 2];
        let scales = &self.scale[row * k / 16..(row + 1) * k / 16];
        (0..k)
            .map(|i| {
                let byte = packed[i / 2];
                let code = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let magnitude = E2M1_VALUES[(code & 0x7) as usize];
                let value = if code & 0x8 != 0 { -magnitude } else { magnitude };
                value * e4m3_to_f32(scales[i / 16]) / self.weight_global_scale
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_decodes() {
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exponent 7, mantissa 0
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0xb8), -1.0);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert!(e4m3_to_f32(0x7f).is_nan());
    }

    #[test]
    fn nvfp4_rows_dequantize() {
        // One row of 32 values: codes 0..15 twice, two blocks with scales 1.0 and 2.0, global scale 0.5.
        let packed: Vec<u8> = (0..16u8).map(|i| ((i * 2 + 1) << 4) | (i * 2)).map(|b| b & 0xff).collect();
        let packed: Vec<u8> = packed.iter().map(|&b| b).collect::<Vec<_>>();
        let codes: Vec<u8> = (0..32).map(|i| (i % 16) as u8).collect();
        let packed: Vec<u8> = codes.chunks(2).map(|pair| pair[0] | (pair[1] << 4)).collect::<Vec<_>>().into_iter().chain(packed.into_iter().take(0)).collect();
        let layer = Nvfp4Linear { packed: &packed, scale: &[0x38, 0x40], weight_global_scale: 0.5, input_global_scale: 1.0, out_features: 1, in_features: 32 };
        let row = layer.row_f32(0);
        let expected: Vec<f32> = codes.iter().enumerate().map(|(i, &c)| {
            let m = E2M1_VALUES[(c & 7) as usize] * if c & 8 != 0 { -1.0 } else { 1.0 };
            m * if i < 16 { 1.0 } else { 2.0 } / 0.5
        }).collect();
        assert_eq!(row, expected);
    }
}
