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
//! - decode_ms: time to read + decode the image
//! - resize_ms: time to resize to target dimensions
//! - tensor_ms: time to convert to NCHW float tensor
//! - preprocess_ms: total preprocess time (decode + resize + tensor)
//! - inference_ms: model inference time (per-image share for GPU batches)
//! - phase: "cpu" or "gpu_batch"

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// Global benchmark log, initialized once per app lifetime.
static BENCH_LOG: std::sync::OnceLock<Mutex<BenchLog>> = std::sync::OnceLock::new();

const CSV_HEADER: &str = "timestamp,file,file_type,file_size_bytes,source_width,source_height,decode_ms,resize_ms,tensor_ms,preprocess_ms,inference_ms,phase";

struct BenchLog {
    path: PathBuf,
}

/// Initialize the benchmark log. Call once during app startup.
/// `dir` is the app data directory where the CSV will be written.
pub fn init(dir: &Path) {
    let csv_path = dir.join("benchmark.csv");

    // Write header if file doesn't exist yet
    let needs_header = !csv_path.exists();
    if needs_header {
        if let Ok(mut f) = File::create(&csv_path) {
            let _ = writeln!(f, "{}", CSV_HEADER);
        }
    }

    let _ = BENCH_LOG.set(Mutex::new(BenchLog { path: csv_path }));
}

/// Start a new scan session — writes a separator and fresh header row
/// so consecutive scans are visually distinct in the CSV.
pub fn begin_scan_session() {
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
    pub resize: Duration,
    pub tensor: Duration,
    pub total: Duration,
}

/// Log a preprocessing + inference result for a single image.
pub fn log_image(timing: &PreprocessTiming, inference: Duration, phase: &str) {
    let Some(log) = BENCH_LOG.get() else { return };
    let Ok(log) = log.lock() else { return };

    let Ok(mut f) = OpenOptions::new().append(true).open(&log.path) else { return };

    let _ = writeln!(
        f,
        "{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{}",
        now_iso(),
        escape_csv(&timing.file),
        timing.file_type,
        timing.file_size_bytes,
        timing.source_width,
        timing.source_height,
        ms(timing.decode),
        ms(timing.resize),
        ms(timing.tensor),
        ms(timing.total),
        ms(inference),
        phase,
    );
}

