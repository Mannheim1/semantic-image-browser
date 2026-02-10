//! Shared image operations used by both embedding and thumbnail modules.
//!
//! This module owns the low-level image primitives:
//! - Decoding images from disk to RGB pixel data
//! - SIMD-accelerated resizing via fast_image_resize

use fast_image_resize::{images::Image as FirImage, ResizeAlg, ResizeOptions, Resizer, PixelType};
use std::path::Path;

/// Decode an image file to RGB pixel data.
/// Uses the fastest available decoder for each format:
/// - JPEG: turbojpeg with scaled decoding
/// - PNG: direct png crate decoder
/// - Other: image crate fallback
///
/// Returns (rgb_data, width, height).
pub fn decode_image_to_rgb(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    if ext == "jpg" || ext == "jpeg" || ext == "jfif" {
        decode_jpeg_scaled(path)
    } else if ext == "png" {
        decode_png(path)
    } else {
        decode_other_format(path)
    }
}

/// Decode a JPEG using turbojpeg with scaled decoding.
/// For large images, decodes at 1/2, 1/4, or 1/8 scale to reduce work.
fn decode_jpeg_scaled(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let jpeg_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read JPEG {}: {}", path.display(), e))?;

    let mut decompressor = turbojpeg::Decompressor::new()
        .map_err(|e| format!("Failed to create JPEG decompressor: {}", e))?;

    // Read header to get original dimensions
    let header = decompressor
        .read_header(&jpeg_data)
        .map_err(|e| format!("Failed to read JPEG header {}: {}", path.display(), e))?;

    // Choose scaling factor based on image size
    // We want the decoded image to be at least IMAGE_SIZE (256) on each dimension
    // but as small as possible to minimize decode and resize work
    let scaling = choose_jpeg_scale(header.width, header.height, 256);
    decompressor.set_scaling_factor(scaling);

    let scaled_header = header.scaled(scaling);
    let scaled_width = scaled_header.width;
    let scaled_height = scaled_header.height;

    // Decode directly to RGB
    let mut rgb_data = vec![0u8; scaled_width * scaled_height * 3];
    let image = turbojpeg::Image {
        pixels: &mut rgb_data[..],
        width: scaled_width,
        pitch: scaled_width * 3,
        height: scaled_height,
        format: turbojpeg::PixelFormat::RGB,
    };

    decompressor
        .decompress(&jpeg_data, image)
        .map_err(|e| format!("Failed to decompress JPEG {}: {}", path.display(), e))?;

    Ok((rgb_data, scaled_width as u32, scaled_height as u32))
}

/// Choose the best JPEG scaling factor for the target size.
/// Returns the largest scale that produces an image >= target_size on both dimensions.
fn choose_jpeg_scale(width: usize, height: usize, target_size: usize) -> turbojpeg::ScalingFactor {
    let min_dim = width.min(height);

    // Try scales from smallest to largest, pick the smallest that still exceeds target
    if min_dim / 8 >= target_size {
        turbojpeg::ScalingFactor::ONE_EIGHTH
    } else if min_dim / 4 >= target_size {
        turbojpeg::ScalingFactor::ONE_QUARTER
    } else if min_dim / 2 >= target_size {
        turbojpeg::ScalingFactor::ONE_HALF
    } else {
        turbojpeg::ScalingFactor::ONE
    }
}

/// Decode a PNG file directly using the png crate, bypassing DynamicImage.
///
/// This avoids the overhead of `image::open()` → `DynamicImage` → `.to_rgb8()`,
/// which allocates intermediate buffers and does format sniffing.
/// Uses `normalize_to_color8()` to guarantee 8-bit output for all PNG variants,
/// then converts to RGB.
fn decode_png(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open PNG {}: {}", path.display(), e))?;
    let reader = std::io::BufReader::new(file);

    let mut decoder = png::Decoder::new(reader);
    // Guarantee 8-bit output: expands palette→RGB, sub-8-bit gray→8-bit, 16-bit→8-bit
    decoder.set_transformations(png::Transformations::normalize_to_color8());

    let mut png_reader = decoder
        .read_info()
        .map_err(|e| format!("Failed to read PNG info {}: {}", path.display(), e))?;

    let buf_size = png_reader
        .output_buffer_size()
        .ok_or_else(|| format!("PNG buffer size overflow: {}", path.display()))?;
    let mut buf = vec![0u8; buf_size];

    let info = png_reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG frame {}: {}", path.display(), e))?;

    let width = info.width;
    let height = info.height;
    let raw = &buf[..info.buffer_size()];

    // Convert to RGB based on the decoded output color type
    let rgb_data = match info.color_type {
        png::ColorType::Rgb => raw.to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in raw.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for &g in &raw[..(width * height) as usize] {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in raw.chunks_exact(2) {
                let g = chunk[0];
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        png::ColorType::Indexed => {
            // With normalize_to_color8(), palette is expanded to Rgb or Rgba.
            // This branch should not be reached.
            return Err(format!(
                "Unexpected indexed color type after expansion in {}",
                path.display()
            ));
        }
    };

    Ok((rgb_data, width, height))
}

/// Decode non-JPEG/non-PNG formats using the image crate.
fn decode_other_format(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image {}: {}", path.display(), e))?;

    let rgb = img.to_rgb8();
    let width = rgb.width();
    let height = rgb.height();
    let rgb_data = rgb.into_raw();

    Ok((rgb_data, width, height))
}

/// Resize RGB image data using fast_image_resize (SIMD-accelerated, bilinear).
/// Optionally accepts a reusable Resizer to avoid per-image allocation in batch processing.
pub fn fast_resize_rgb(
    rgb_data: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    resizer: Option<&mut Resizer>,
) -> Result<Vec<u8>, String> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err("Invalid image dimensions".to_string());
    }

    // If no resize needed, return original
    if src_width == dst_width && src_height == dst_height {
        return Ok(rgb_data.to_vec());
    }

    let mut rgb_copy = rgb_data.to_vec();
    let src_image = FirImage::from_slice_u8(
        src_width,
        src_height,
        &mut rgb_copy,
        PixelType::U8x3,
    )
    .map_err(|e| format!("Failed to create source image: {}", e))?;

    let mut dst_image = FirImage::new(dst_width, dst_height, PixelType::U8x3);

    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(
        fast_image_resize::FilterType::Bilinear,
    ));

    let mut fallback_resizer;
    let resizer = match resizer {
        Some(r) => r,
        None => {
            fallback_resizer = Resizer::new();
            &mut fallback_resizer
        }
    };

    resizer
        .resize(&src_image, &mut dst_image, Some(&options))
        .map_err(|e| format!("Failed to resize image: {}", e))?;

    Ok(dst_image.into_vec())
}
