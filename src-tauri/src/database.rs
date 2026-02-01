//! Database layer for semantic image browser.
//!
//! # SQL Injection Prevention
//!
//! This module uses LanceDB's SQL-like query language for filtering operations.
//! As of LanceDB v0.23.1, parameterized queries are not yet supported
//! (see: https://github.com/lancedb/lancedb/issues/1368).
//!
//! To prevent SQL injection vulnerabilities, all user-provided string values
//! are escaped using the `escape_sql_string()` function before being interpolated
//! into SQL predicates. This function:
//! - Doubles single quotes (SQL standard escaping)
//! - Doubles backslashes (to prevent escape sequence attacks)
//! - Rejects null bytes (invalid in SQL strings)
//!
//! **TODO**: Once LanceDB adds parameterized query support, migrate to using
//! native parameters instead of manual escaping.

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, RecordBatch, RecordBatchIterator, StringArray,
    TimestampMillisecondArray, UInt64Array,
};
use std::collections::HashMap;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Manager};

use crate::config::AppConfig;

pub const TABLE_NAME: &str = "images";
pub const VISUAL_EMBEDDING_DIM: i32 = 768; // SigLIP2-base pooler_output dimension
pub const OCR_EMBEDDING_DIM: i32 = 384;

/// Number of embedding model slots available.
/// This allows storing embeddings from multiple models without schema changes.
/// Currently only slot 1 is used for search, but others can be populated
/// to allow switching models without recalculating embeddings.
pub const NUM_EMBEDDING_SLOTS: usize = 5;

/// Escapes a string value for safe use in LanceDB SQL predicates.
///
/// LanceDB does not yet support parameterized queries (see lancedb/lancedb#1368).
/// This function provides comprehensive escaping for string values used in SQL
/// predicates like `only_if()` and `delete()`.
///
/// Escaping rules:
/// - Single quotes (') are doubled ('')
/// - Backslashes (\) are doubled (\\)
/// - Null bytes are rejected (not allowed in SQL strings)
///
/// This prevents SQL injection while we wait for native parameter support.
fn escape_sql_string(s: &str) -> Result<String, String> {
    // Reject null bytes - they're not valid in SQL strings and could indicate
    // an attack attempting to truncate the string
    if s.contains('\0') {
        return Err("Path contains null byte".to_string());
    }

    // Escape single quotes (SQL standard) and backslashes
    let escaped = s.replace('\\', "\\\\").replace('\'', "''");

    Ok(escaped)
}

pub struct ImageRecord {
    pub path: String,
    pub file_type: String,
    pub file_size: u64,
    pub created_at: i64,
    pub modified_at: i64,
    /// Visual embedding for slot 1 (currently the only slot used for indexing/search).
    /// Other slots (2-5) exist in the schema for future model switching support.
    pub visual_embedding: Option<Vec<f32>>,
    /// Model ID for slot 1 (e.g., "siglip2-base-patch16-256").
    /// Used to identify which model generated the embedding.
    pub model_id: Option<String>,
}

pub fn db_path(app: &AppHandle, _config: &AppConfig) -> Result<PathBuf, String> {
    let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    Ok(app_data.join("lancedb"))
}

pub fn create_schema() -> Arc<Schema> {
    // Schema supports 5 embedding model slots to allow switching models without
    // recalculating all embeddings. Currently only slot 1 is used for search.
    // Each slot has: model_id_N (string) and embedding_N (768-dim vector).
    let mut fields = vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("file_type", DataType::Utf8, false),
        Field::new("file_size", DataType::UInt64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new(
            "modified_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
    ];

    // Add 5 embedding slots (model_id + embedding pairs)
    for i in 1..=NUM_EMBEDDING_SLOTS {
        fields.push(Field::new(format!("model_id_{}", i), DataType::Utf8, true));
        fields.push(Field::new(
            format!("embedding_{}", i),
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VISUAL_EMBEDDING_DIM,
            ),
            true,
        ));
    }

    // OCR fields (unchanged)
    fields.push(Field::new("ocr_text", DataType::Utf8, true));
    fields.push(Field::new(
        "ocr_embedding",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            OCR_EMBEDDING_DIM,
        ),
        true,
    ));

    Arc::new(Schema::new(fields))
}

pub async fn open_connection(app: &AppHandle, config: &AppConfig) -> Result<Connection, String> {
    let path = db_path(app, config)?;
    connect(path.to_string_lossy().as_ref())
        .execute()
        .await
        .map_err(|e| e.to_string())
}

pub async fn get_or_create_table(db: &Connection) -> Result<Table, String> {
    let tables = db.table_names().execute().await.map_err(|e| e.to_string())?;

    if tables.contains(&TABLE_NAME.to_string()) {
        db.open_table(TABLE_NAME)
            .execute()
            .await
            .map_err(|e| e.to_string())
    } else {
        let schema = create_schema();
        db.create_empty_table(TABLE_NAME, schema)
            .execute()
            .await
            .map_err(|e| e.to_string())
    }
}

fn create_null_embedding_array(dim: i32, len: usize) -> ArrayRef {
    use arrow_array::builder::FixedSizeListBuilder;
    use arrow_array::builder::Float32Builder;

    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
    for _ in 0..len {
        // Must append dim values before marking entry as null
        for _ in 0..dim {
            builder.values().append_null();
        }
        builder.append(false); // false = null entry
    }
    Arc::new(builder.finish())
}

/// Create an embedding array from a list of optional embeddings.
/// Each embedding is either Some(Vec<f32>) or None (null).
fn create_embedding_array(dim: i32, embeddings: &[Option<&Vec<f32>>]) -> ArrayRef {
    use arrow_array::builder::FixedSizeListBuilder;
    use arrow_array::builder::Float32Builder;

    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
    for emb in embeddings {
        match emb {
            Some(values) => {
                for &v in values.iter() {
                    builder.values().append_value(v);
                }
                builder.append(true); // true = valid entry
            }
            None => {
                for _ in 0..dim {
                    builder.values().append_null();
                }
                builder.append(false); // false = null entry
            }
        }
    }
    Arc::new(builder.finish())
}

pub async fn upsert_images(table: &Table, records: Vec<ImageRecord>) -> Result<(), String> {
    if records.is_empty() {
        return Ok(());
    }

    let len = records.len();
    let paths: Vec<&str> = records.iter().map(|r| r.path.as_str()).collect();
    let file_types: Vec<&str> = records.iter().map(|r| r.file_type.as_str()).collect();
    let file_sizes: Vec<u64> = records.iter().map(|r| r.file_size).collect();
    let created_ats: Vec<i64> = records.iter().map(|r| r.created_at).collect();
    let modified_ats: Vec<i64> = records.iter().map(|r| r.modified_at).collect();

    // Slot 1 embeddings and model IDs (currently the only slot used)
    let model_ids_1: Vec<Option<&str>> = records
        .iter()
        .map(|r| r.model_id.as_deref())
        .collect();
    let embeddings_1: Vec<Option<&Vec<f32>>> = records
        .iter()
        .map(|r| r.visual_embedding.as_ref())
        .collect();

    let schema = create_schema();

    // Build columns in schema order:
    // path, file_type, file_size, created_at, modified_at,
    // model_id_1, embedding_1, model_id_2, embedding_2, ... model_id_5, embedding_5,
    // ocr_text, ocr_embedding
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(paths)),
        Arc::new(StringArray::from(file_types)),
        Arc::new(UInt64Array::from(file_sizes)),
        Arc::new(TimestampMillisecondArray::from(created_ats)),
        Arc::new(TimestampMillisecondArray::from(modified_ats)),
    ];

    // Add embedding slots 1-5
    // Slot 1 uses actual data; slots 2-5 are null (reserved for future model switching)
    for slot in 1..=NUM_EMBEDDING_SLOTS {
        if slot == 1 {
            columns.push(Arc::new(StringArray::from(model_ids_1.clone())));
            columns.push(create_embedding_array(VISUAL_EMBEDDING_DIM, &embeddings_1));
        } else {
            // Slots 2-5: null for now (reserved for future use)
            columns.push(Arc::new(StringArray::from(vec![None::<&str>; len])));
            columns.push(create_null_embedding_array(VISUAL_EMBEDDING_DIM, len));
        }
    }

    // OCR fields (null for now)
    columns.push(Arc::new(StringArray::from(vec![None::<&str>; len])));
    columns.push(create_null_embedding_array(OCR_EMBEDDING_DIM, len));

    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| e.to_string())?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

    let mut merge = table.merge_insert(&["path"]);
    merge
        .when_matched_update_all(None)
        .when_not_matched_insert_all();
    merge
        .execute(Box::new(batches))
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub async fn remove_images(table: &Table, paths: &[String]) -> Result<(), String> {
    for path in paths {
        let escaped = escape_sql_string(path)?;
        table
            .delete(&format!("path = '{}'", escaped))
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub async fn get_all_paths(table: &Table) -> Result<Vec<String>, String> {
    let batches: Vec<RecordBatch> = table
        .query()
        .select(lancedb::query::Select::Columns(vec!["path".to_string()]))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?
        .try_collect()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;

    let mut paths = Vec::new();
    for batch in batches {
        let path_col = batch
            .column_by_name("path")
            .ok_or("path column not found")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("path column is not a string array")?;

        for i in 0..path_col.len() {
            if !path_col.is_null(i) {
                paths.push(path_col.value(i).to_string());
            }
        }
    }

    Ok(paths)
}

/// Returns a map of path -> modified_at for all images in the table.
pub async fn get_all_modified_times(table: &Table) -> Result<HashMap<String, i64>, String> {
    let batches: Vec<RecordBatch> = table
        .query()
        .select(lancedb::query::Select::Columns(vec![
            "path".to_string(),
            "modified_at".to_string(),
        ]))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?
        .try_collect()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;

    let mut map = HashMap::new();
    for batch in batches {
        let path_col = batch
            .column_by_name("path")
            .ok_or("path column not found")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("path column is not a string array")?;
        let modified_col = batch
            .column_by_name("modified_at")
            .ok_or("modified_at column not found")?
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or("modified_at column is not a timestamp array")?;

        for i in 0..path_col.len() {
            if !path_col.is_null(i) {
                map.insert(path_col.value(i).to_string(), modified_col.value(i));
            }
        }
    }

    Ok(map)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageInfo {
    pub path: String,
    pub file_type: String,
    pub file_size: u64,
    pub created_at: i64,
    pub modified_at: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub path: String,
    pub file_type: String,
    pub file_size: u64,
    pub created_at: i64,
    pub modified_at: i64,
    pub sort_score: Option<f32>,
}

pub async fn get_all_images(table: &Table) -> Result<Vec<ImageInfo>, String> {
    let batches: Vec<RecordBatch> = table
        .query()
        .select(lancedb::query::Select::Columns(vec![
            "path".to_string(),
            "file_type".to_string(),
            "file_size".to_string(),
            "created_at".to_string(),
            "modified_at".to_string(),
        ]))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?
        .try_collect()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;

    let mut images = Vec::new();
    for batch in batches {
        let path_col = batch.column_by_name("path").ok_or("path not found")?
            .as_any().downcast_ref::<StringArray>().ok_or("path not string")?;
        let file_type_col = batch.column_by_name("file_type").ok_or("file_type not found")?
            .as_any().downcast_ref::<StringArray>().ok_or("file_type not string")?;
        let file_size_col = batch.column_by_name("file_size").ok_or("file_size not found")?
            .as_any().downcast_ref::<UInt64Array>().ok_or("file_size not u64")?;
        let created_col = batch.column_by_name("created_at").ok_or("created_at not found")?
            .as_any().downcast_ref::<TimestampMillisecondArray>().ok_or("created_at not ts")?;
        let modified_col = batch.column_by_name("modified_at").ok_or("modified_at not found")?
            .as_any().downcast_ref::<TimestampMillisecondArray>().ok_or("modified_at not ts")?;

        for i in 0..batch.num_rows() {
            images.push(ImageInfo {
                path: path_col.value(i).to_string(),
                file_type: file_type_col.value(i).to_string(),
                file_size: file_size_col.value(i),
                created_at: created_col.value(i),
                modified_at: modified_col.value(i),
            });
        }
    }

    Ok(images)
}

fn extract_search_results_from_batches(batches: Vec<RecordBatch>) -> Result<Vec<SearchResult>, String> {
    let mut images = Vec::new();
    for batch in batches {
        let path_col = batch.column_by_name("path").ok_or("path not found")?
            .as_any().downcast_ref::<StringArray>().ok_or("path not string")?;
        let file_type_col = batch.column_by_name("file_type").ok_or("file_type not found")?
            .as_any().downcast_ref::<StringArray>().ok_or("file_type not string")?;
        let file_size_col = batch.column_by_name("file_size").ok_or("file_size not found")?
            .as_any().downcast_ref::<UInt64Array>().ok_or("file_size not u64")?;
        let created_col = batch.column_by_name("created_at").ok_or("created_at not found")?
            .as_any().downcast_ref::<TimestampMillisecondArray>().ok_or("created_at not ts")?;
        let modified_col = batch.column_by_name("modified_at").ok_or("modified_at not found")?
            .as_any().downcast_ref::<TimestampMillisecondArray>().ok_or("modified_at not ts")?;
        let distance_col = batch.column_by_name("_distance");

        for i in 0..batch.num_rows() {
            let sort_score = distance_col.and_then(|col| {
                if col.is_null(i) {
                    return None;
                }
                if let Some(col) = col.as_any().downcast_ref::<Float32Array>() {
                    return Some(col.value(i));
                }
                if let Some(col) = col.as_any().downcast_ref::<Float64Array>() {
                    return Some(col.value(i) as f32);
                }
                None
            });

            images.push(SearchResult {
                path: path_col.value(i).to_string(),
                file_type: file_type_col.value(i).to_string(),
                file_size: file_size_col.value(i),
                created_at: created_col.value(i),
                modified_at: modified_col.value(i),
                sort_score,
            });
        }
    }
    Ok(images)
}

pub async fn search_by_filename(table: &Table, query: &str) -> Result<Vec<SearchResult>, String> {
    let escaped = escape_sql_string(query)?;
    let pattern = format!("%{}%", escaped);

    let batches: Vec<RecordBatch> = table
        .query()
        .only_if(format!("path LIKE '{}' ", pattern))
        .select(lancedb::query::Select::Columns(vec![
            "path".to_string(),
            "file_type".to_string(),
            "file_size".to_string(),
            "created_at".to_string(),
            "modified_at".to_string(),
        ]))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?
        .try_collect()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;

    extract_search_results_from_batches(batches)
}

/// Search for images using vector similarity on embedding slot 1.
/// Returns images ordered by similarity to the query embedding (most similar first).
/// Only searches images that have an embedding in slot 1.
///
/// Note: Currently only slot 1 is used for search. Slots 2-5 exist in the schema
/// to allow switching between models without recalculating embeddings, but
/// searching those slots is not yet implemented.
pub async fn search_by_embedding(
    table: &Table,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<SearchResult>, String> {
    use lancedb::query::QueryBase;

    // Search using embedding_1 (slot 1) - the only slot currently used
    let batches: Vec<RecordBatch> = table
        .vector_search(query_embedding.to_vec())
        .map_err(|e| format!("Failed to create vector search: {}", e))?
        .column("embedding_1")
        .limit(limit)
        .select(lancedb::query::Select::Columns(vec![
            "path".to_string(),
            "file_type".to_string(),
            "file_size".to_string(),
            "created_at".to_string(),
            "modified_at".to_string(),
        ]))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?
        .try_collect()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;

    extract_search_results_from_batches(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_sql_string_basic() {
        assert_eq!(
            escape_sql_string("normal_path.jpg").unwrap(),
            "normal_path.jpg"
        );
    }

    #[test]
    fn test_escape_sql_string_single_quote() {
        assert_eq!(
            escape_sql_string("image's_file.jpg").unwrap(),
            "image''s_file.jpg"
        );
    }

    #[test]
    fn test_escape_sql_string_backslash() {
        assert_eq!(
            escape_sql_string(r"C:\Users\test.jpg").unwrap(),
            r"C:\\Users\\test.jpg"
        );
    }

    #[test]
    fn test_escape_sql_string_both() {
        assert_eq!(
            escape_sql_string(r"C:\User's\test.jpg").unwrap(),
            r"C:\\User''s\\test.jpg"
        );
    }

    #[test]
    fn test_escape_sql_string_sql_injection_attempt() {
        // Try to inject SQL: ' OR '1'='1
        assert_eq!(
            escape_sql_string("test' OR '1'='1.jpg").unwrap(),
            "test'' OR ''1''=''1.jpg"
        );
    }

    #[test]
    fn test_escape_sql_string_null_byte() {
        // Null bytes should be rejected
        assert!(escape_sql_string("test\0.jpg").is_err());
    }

    #[test]
    fn test_escape_sql_string_unicode() {
        // Unicode should pass through unchanged
        assert_eq!(
            escape_sql_string("图片_文件.jpg").unwrap(),
            "图片_文件.jpg"
        );
    }
}
