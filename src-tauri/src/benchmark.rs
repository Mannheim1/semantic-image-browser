//! Benchmark logging for scan performance analysis.
//!
//! Writes a CSV file to the app data directory with per-image timing data.
//! Each scan session appends rows, separated by a blank line and header for clarity.
//!
//! CSV columns:
//! - timestamp: ISO 8601 wall-clock time
//! - file: filename (not full path, for readability)
//! - file_type: extension (jpg, png, etc.)
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

/// Global benchmark log, initialized once per app lifetime.
static BENCH_LOG: std::sync::OnceLock<Mutex<BenchLog>> = std::sync::OnceLock::new();

/// Runtime toggle — when false, `log_image` and `begin_scan_session` are no-ops.
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}


const CSV_HEADER: &str = "timestamp,file,file_type,file_size_bytes,source_width,source_height,phase,decode_ms,decode_ms_per_kb,decode_ms_per_kpx,thumbnail_ms,thumbnail_ms_per_kpx,resize_ms,resize_ms_per_kpx,tensor_ms,inference_ms";

struct BenchLog {
    path: PathBuf,
}

/// Initialize the benchmark log. Call once during app startup.
/// `dir` is the app data directory where the CSV will be written.
pub fn init(dir: &Path) {
    let csv_path = dir.join("benchmark.csv");

    // Overwrite on each app launch — begin_scan_session writes headers per scan
    if let Ok(mut f) = File::create(&csv_path) {
        let _ = writeln!(f, "{}", CSV_HEADER);
    }

    let _ = BENCH_LOG.set(Mutex::new(BenchLog { path: csv_path }));
}

/// Start a new scan session — writes a separator and fresh header row
/// so consecutive scans are visually distinct in the CSV.
pub fn begin_scan_session() {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let Some(log) = BENCH_LOG.get() else { return };
    let Ok(log) = log.lock() else { return };

    let Ok(mut f) = OpenOptions::new().append(true).open(&log.path) else { return };
    let _ = writeln!(f); // blank line separator
    let _ = writeln!(f, "{}", CSV_HEADER);
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
    let Some(log) = BENCH_LOG.get() else { return };
    let Ok(log) = log.lock() else { return };

    let Ok(mut f) = OpenOptions::new().append(true).open(&log.path) else { return };

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

