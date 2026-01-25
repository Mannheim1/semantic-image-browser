use lancedb::connect;
use ort::session::Session;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn test_lancedb() -> Result<String, String> {
    let db = connect("data/lancedb").execute().await.map_err(|e| e.to_string())?;
    let tables = db.table_names().execute().await.map_err(|e| e.to_string())?;
    Ok(format!("LanceDB connected! Tables: {:?}", tables))
}

#[tauri::command]
fn test_onnx() -> Result<String, String> {
    let _builder = Session::builder().map_err(|e| e.to_string())?;
    Ok("ONNX Runtime initialized successfully!".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, test_lancedb, test_onnx])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
