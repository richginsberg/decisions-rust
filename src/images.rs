//! Image checks and preprocessing: the Qwen2-VL image processor's `smart_resize`, normalization and patch
//! layout, so a request's images take the same positions and give the backbone the same pixels as the Python
//! processor (image_processing_qwen2_vl.py).

use base64::Engine as _;
use image::GenericImageView;

use crate::model::Model;
use crate::protocol::ImageGrid;

/// decider_api.py's checks: decodable PNG, JPEG or WebP of at most 16 million pixels.
pub fn check(content: &[u8]) -> Result<(), String> {
    let format = image::guess_format(content).map_err(|_| "Invalid image data.".to_string())?;
    if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP) {
        return Err("Unsupported image format.".into());
    }
    let reader = image::ImageReader::with_format(std::io::Cursor::new(content), format);
    let (width, height) = reader.into_dimensions().map_err(|_| "Invalid image data.".to_string())?;
    if width as u64 * height as u64 > 16_000_000 {
        return Err("Each image must have at most 16 million pixels.".into());
    }
    Ok(())
}

/// The processor's `smart_resize`: height and width rounded to multiples of `factor`, scaled to keep the pixel
/// count within [min_pixels, max_pixels], with the aspect ratio capped at 200.
pub fn smart_resize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) -> Result<(u32, u32), String> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!("absolute aspect ratio must be smaller than 200, got {:.1}", h.max(w) / h.min(w)));
    }
    let factor_f = factor as f64;
    // Python's round() rounds halves to even
    let round = |v: f64| ((v / factor_f).round_ties_even() * factor_f) as u64;
    let mut h_bar = round(h).max(factor as u64);
    let mut w_bar = round(w).max(factor as u64);
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = ((h / beta / factor_f).floor() * factor_f) as u64;
        w_bar = ((w / beta / factor_f).floor() * factor_f) as u64;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / factor_f).ceil() * factor_f) as u64;
        w_bar = ((w * beta / factor_f).ceil() * factor_f) as u64;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// An image decoded, resized, normalized and laid out as the vision tower's patches.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    pub grid: ImageGrid,
    /// [grid.t * grid.h * grid.w, channels * temporal_patch * patch * patch] in the processor's patch order.
    pub patches: Vec<f32>,
}

pub struct ImageSettings {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Decode a data URL and preprocess it as image_processing_qwen2_vl.py's `_preprocess` does for one image:
/// convert to RGB, bicubic resize to the smart size, rescale by 1/255, normalize, repeat the frame to the
/// temporal patch size, and flatten into patches ordered (t, h/merge, w/merge, merge, merge).
pub fn prepare(data_url: &str, settings: &ImageSettings) -> Result<PreparedImage, String> {
    let (_, encoded) = data_url.split_once(',').ok_or("Images must be base64 data URLs.")?;
    let content = base64::engine::general_purpose::STANDARD.decode(encoded).map_err(|_| "Invalid image data.")?;
    let decoded = image::load_from_memory(&content).map_err(|_| "Invalid image data.")?;
    let (width, height) = decoded.dimensions();
    let factor = (settings.patch_size * settings.merge_size) as u32;
    let (resized_h, resized_w) = smart_resize(height, width, factor, settings.min_pixels, settings.max_pixels)?;
    let rgb = decoded.to_rgb8();
    let resized = image::imageops::resize(&rgb, resized_w, resized_h, image::imageops::FilterType::CatmullRom);
    let (p, m, tp) = (settings.patch_size, settings.merge_size, settings.temporal_patch_size);
    let grid = ImageGrid { t: 1, h: resized_h as usize / p, w: resized_w as usize / p };
    let patch_len = 3 * tp * p * p;
    let mut patches = vec![0f32; grid.t * grid.h * grid.w * patch_len];
    // Patch order: (t, h/m, w/m, m, m); within a patch: (channel, temporal, py, px).
    let (hb, wb) = (grid.h / m, grid.w / m);
    for bh in 0..hb {
        for bw in 0..wb {
            for mh in 0..m {
                for mw in 0..m {
                    let patch_index = ((bh * wb + bw) * m + mh) * m + mw;
                    let (y0, x0) = ((bh * m + mh) * p, (bw * m + mw) * p);
                    let out = &mut patches[patch_index * patch_len..(patch_index + 1) * patch_len];
                    for c in 0..3 {
                        for t in 0..tp {
                            for py in 0..p {
                                for px in 0..p {
                                    let pixel = resized.get_pixel((x0 + px) as u32, (y0 + py) as u32)[c] as f32;
                                    out[((c * tp + t) * p + py) * p + px] = (pixel / 255.0 - settings.mean[c]) / settings.std[c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(PreparedImage { grid, patches })
}

/// Every image of a request prepared once: the encoder takes the grids, the model's pass the pixels.
pub fn prepare_all(images: &[String], model: &dyn Model) -> Result<Vec<std::sync::Arc<PreparedImage>>, String> {
    let settings = model.image_settings();
    images.iter().map(|url| prepare(url, &settings).map(std::sync::Arc::new)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_the_processor() {
        // 1920x1080 with factor 32 and the checkpoint's bounds: rounding to multiples of 32 gives 1088x1920, which is
        // over max pixels, so both sides are scaled down by sqrt(1080*1920 / 1_638_400) and floored to multiples of 32.
        assert_eq!(smart_resize(1080, 1920, 32, 65_536, 1_638_400).unwrap(), (960, 1696));
        // A small image is scaled up to the minimum pixel count.
        let (h, w) = smart_resize(100, 100, 32, 65_536, 1_638_400).unwrap();
        assert!(h as u64 * w as u64 >= 65_536 && h % 32 == 0 && w % 32 == 0);
        // A huge image is scaled down below the maximum.
        let (h, w) = smart_resize(4000, 4000, 32, 65_536, 1_638_400).unwrap();
        assert!(h as u64 * w as u64 <= 1_638_400 && h % 32 == 0);
        assert!(smart_resize(10, 5000, 32, 65_536, 1_638_400).is_err());
    }
}
