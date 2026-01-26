use base64::{engine::general_purpose::STANDARD, Engine};
use std::path::Path;

#[cfg(windows)]
use windows::{
    core::HSTRING,
    Win32::{
        Graphics::Gdi::{
            CreateCompatibleDC, DeleteDC, DeleteObject, GetDIBits, SelectObject, BITMAPINFO,
            BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
        },
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED},
        UI::Shell::{IShellItemImageFactory, SHCreateItemFromParsingName, SIIGBF_RESIZETOFIT},
    },
};

#[cfg(windows)]
pub fn get_thumbnail_base64(path: &Path, requested_size: u32) -> Result<String, String> {
    unsafe {
        // Initialize COM for this thread
        let com_initialized = CoInitializeEx(None, COINIT_APARTMENTTHREADED).is_ok();

        let result = get_thumbnail_internal(path, requested_size);

        // Uninitialize COM if we initialized it
        if com_initialized {
            CoUninitialize();
        }

        result
    }
}

#[cfg(windows)]
unsafe fn get_thumbnail_internal(path: &Path, requested_size: u32) -> Result<String, String> {
    // Convert path to wide string
    let path_str = path
        .to_str()
        .ok_or_else(|| "Invalid path encoding".to_string())?;
    let wide_path = HSTRING::from(path_str);

    // Create IShellItem from path
    let shell_item: IShellItemImageFactory = SHCreateItemFromParsingName(&wide_path, None)
        .map_err(|e| format!("Failed to create shell item for '{}': {}", path.display(), e))?;

    // Get thumbnail as HBITMAP
    let size = windows::Win32::Foundation::SIZE {
        cx: requested_size as i32,
        cy: requested_size as i32,
    };

    let hbitmap = shell_item
        .GetImage(size, SIIGBF_RESIZETOFIT)
        .map_err(|e| format!("Failed to get thumbnail for '{}': {}", path.display(), e))?;

    // Convert HBITMAP to PNG bytes
    let png_bytes = hbitmap_to_png(hbitmap, requested_size)?;

    // Clean up GDI object
    let _ = DeleteObject(hbitmap.into());

    // Return as base64 data URL
    let base64_data = STANDARD.encode(&png_bytes);
    Ok(format!("data:image/png;base64,{}", base64_data))
}

#[cfg(windows)]
unsafe fn hbitmap_to_png(
    hbitmap: windows::Win32::Graphics::Gdi::HBITMAP,
    _size: u32,
) -> Result<Vec<u8>, String> {
    use windows::Win32::Graphics::Gdi::{GetObjectW, BITMAP};

    // Get bitmap dimensions
    let mut bitmap = BITMAP::default();
    let result = GetObjectW(
        hbitmap.into(),
        std::mem::size_of::<BITMAP>() as i32,
        Some(&mut bitmap as *mut _ as *mut _),
    );
    if result == 0 {
        return Err("Failed to get bitmap info".to_string());
    }

    let width = bitmap.bmWidth as u32;
    let height = bitmap.bmHeight as u32;

    // Create compatible DC
    let hdc = CreateCompatibleDC(None);
    if hdc.is_invalid() {
        return Err("Failed to create compatible DC".to_string());
    }

    // Select bitmap into DC
    let old_bitmap = SelectObject(hdc, hbitmap.into());

    // Set up BITMAPINFO for 32-bit BGRA
    let mut bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width as i32,
            biHeight: -(height as i32), // Negative for top-down DIB
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: [Default::default()],
    };

    // Allocate buffer for pixel data (BGRA)
    let mut pixels: Vec<u8> = vec![0u8; (width * height * 4) as usize];

    // Get the bits
    let lines = GetDIBits(
        hdc,
        hbitmap,
        0,
        height,
        Some(pixels.as_mut_ptr() as *mut _),
        &mut bmi,
        DIB_RGB_COLORS,
    );

    // Clean up GDI resources
    SelectObject(hdc, old_bitmap);
    let _ = DeleteDC(hdc);

    if lines == 0 {
        return Err("Failed to get DIB bits".to_string());
    }

    // Convert BGRA to RGBA
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2); // Swap B and R
    }

    // Encode to PNG using the image crate
    let img =
        image::RgbaImage::from_raw(width, height, pixels).ok_or("Failed to create image buffer")?;

    let mut png_bytes: Vec<u8> = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )
    .map_err(|e| format!("Failed to encode PNG: {}", e))?;

    Ok(png_bytes)
}

#[cfg(not(windows))]
pub fn get_thumbnail_base64(_path: &Path, _requested_size: u32) -> Result<String, String> {
    Err("Thumbnail generation is only supported on Windows".to_string())
}
