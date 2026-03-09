//! Benchmark logging for scan performance analysis.
//!
//! Each scan session creates a new CSV file in the scanned directory:
//! `benchmark_[backend]_yyyymmdd-hhmmss.csv` (local time).
//!
//! CSV columns:
//! - timestamp: ISO 8601 wall-clock time
//! - file: filename (not full path, for readability)
//! - file_type: detected format via magic bytes (jpg, png, webp, etc.)
//! - file_size_bytes: raw file size on disk
//! - source_width: decoded image width in pixels
//! - source_height: decoded image height in pixels
//! - phase: "cpu" or "gpu_batch"
//! - decode_ms: time to read + decode the image
//! - decode_ms_per_kb: decode time normalized by file size
//! - decode_ms_per_kpx: decode time normalized by pixel count
//! - thumbnail_ms: time to resize + encode + write thumbnail
//! - thumbnail_ms_per_kpx: thumbnail time normalized by pixel count
//! - resize_ms: time to resize to embedding target dimensions
//! - resize_ms_per_kpx: resize time normalized by pixel count
//! - tensor_ms: time to convert to NCHW float tensor
//! - inference_ms: model inference time (per-image share for GPU batches)

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Runtime toggle — when false, `log_image` and `begin_scan_session` are no-ops.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Directory where benchmark CSVs are written (app data dir), set once at startup.
static OUTPUT_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Path to the CSV file for the current scan session.
static CURRENT_CSV: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}


const CSV_HEADER: &str = "timestamp,file,file_type,file_size_bytes,source_width,source_height,phase,decode_ms,decode_ms_per_kb,decode_ms_per_kpx,thumbnail_ms,thumbnail_ms_per_kpx,resize_ms,resize_ms_per_kpx,tensor_ms,inference_ms";

fn backend_type() -> &'static str {
    #[cfg(feature = "backend-cuda")]
    { "cuda" }
    #[cfg(feature = "backend-coreml")]
    { "coreml" }
    #[cfg(all(not(feature = "backend-cuda"), not(feature = "backend-coreml")))]
    { "cpu" }
}

/// Format the current local time as `yyyymmdd-hhmmss` for the CSV filename.
#[cfg(unix)]
fn local_now_filename() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as libc::time_t;

    let mut tm = std::mem::MaybeUninit::<libc::tm>::uninit();
    unsafe {
        libc::localtime_r(&secs, tm.as_mut_ptr());
        let tm = tm.assume_init();
        format!(
            "{:04}{:02}{:02}-{:02}{:02}{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
        )
    }
}

#[cfg(not(unix))]
fn local_now_filename() -> String {
    // Fallback: use UTC on non-Unix platforms
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;

    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year { break; }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days: [i64; 12] = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0;
    for &md in &month_days {
        if remaining_days < md { break; }
        remaining_days -= md;
        m += 1;
    }

    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        y, m + 1, remaining_days + 1,
        time_of_day / 3600, (time_of_day % 3600) / 60, time_of_day % 60,
    )
}

/// Initialize the benchmark output directory. Call once during app startup.
pub fn init(dir: &Path) {
    let _ = OUTPUT_DIR.set(dir.to_path_buf());
}

/// Start a new scan session — creates a fresh CSV in the app data directory.
pub fn begin_scan_session() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let Some(dir) = OUTPUT_DIR.get() else { return };

    let filename = format!("benchmark_{}_{}.csv", backend_type(), local_now_filename());
    let csv_path = dir.join(filename);

    if let Ok(mut f) = File::create(&csv_path) {
        let _ = writeln!(f, "{}", CSV_HEADER);
    }

    if let Ok(mut current) = CURRENT_CSV.lock() {
        *current = Some(csv_path);
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn now_iso() -> String {
    // Simple timestamp: YYYY-MM-DD HH:MM:SS
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Convert to rough UTC components (no TZ library needed for benchmarking)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since epoch to Y-M-D (simplified, correct for 2000-2099)
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days: [i64; 12] = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0;
    for &md in &month_days {
        if remaining_days < md {
            break;
        }
        remaining_days -= md;
        m += 1;
    }

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        y, m + 1, remaining_days + 1, hours, minutes, seconds
    )
}

/// Timing data collected during image preprocessing.
#[derive(Clone)]
pub struct PreprocessTiming {
    pub file: String,
    pub file_type: String,
    pub file_size_bytes: u64,
    pub source_width: u32,
    pub source_height: u32,
    pub decode: Duration,
    pub thumbnail: Duration,
    pub resize: Duration,
    pub tensor: Duration,
}

/// Log a preprocessing + inference result for a single image.
pub fn log_image(timing: &PreprocessTiming, inference: Duration, phase: &str) {
    if !ENABLED.load(Ordering::Relaxed) { return; }

    let path = {
        let Ok(guard) = CURRENT_CSV.lock() else { return };
        guard.clone()
    };
    let Some(path) = path else { return };

    let Ok(mut f) = OpenOptions::new().append(true).open(&path) else { return };

    let kb = timing.file_size_bytes as f64 / 1024.0;
    let kpx = (timing.source_width as f64 * timing.source_height as f64) / 1000.0;

    let decode = ms(timing.decode);
    let thumbnail = ms(timing.thumbnail);
    let resize = ms(timing.resize);

    let _ = writeln!(
        f,
        "{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
        now_iso(),
        escape_csv(&timing.file),
        timing.file_type,
        timing.file_size_bytes,
        timing.source_width,
        timing.source_height,
        phase,
        decode,
        decode / kb,
        decode / kpx,
        thumbnail,
        thumbnail / kpx,
        resize,
        resize / kpx,
        ms(timing.tensor),
        ms(inference),
    );
}
