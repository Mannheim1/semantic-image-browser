use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    TimestampMillisecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Manager};

use crate::config::AppConfig;

pub const TABLE_NAME: &str = "images";
pub const VISUAL_EMBEDDING_DIM: i32 = 512;
pub const OCR_EMBEDDING_DIM: i32 = 384;

pub struct ImageRecord {
    pub path: String,
    pub file_type: String,
    pub file_size: i64,
    pub created_at: i64,
    pub modified_at: i64,
}

pub fn db_path(app: &AppHandle, config: &AppConfig) -> Result<PathBuf, String> {
    match &config.custom_data_location {
        Some(custom) => Ok(PathBuf::from(custom).join("lancedb")),
        None => {
            let app_data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
            Ok(app_data.join("lancedb"))
        }
    }
}

pub fn create_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new(
            "visual_embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VISUAL_EMBEDDING_DIM,
            ),
            true,
        ),
        Field::new("ocr_text", DataType::Utf8, true),
        Field::new(
            "ocr_embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                OCR_EMBEDDING_DIM,
            ),
            true,
        ),
        Field::new("file_type", DataType::Utf8, false),
        Field::new("file_size", DataType::Int64, false),
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
    ]))
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

pub async fn upsert_images(table: &Table, records: Vec<ImageRecord>) -> Result<(), String> {
    if records.is_empty() {
        return Ok(());
    }

    let len = records.len();
    let paths: Vec<&str> = records.iter().map(|r| r.path.as_str()).collect();
    let file_types: Vec<&str> = records.iter().map(|r| r.file_type.as_str()).collect();
    let file_sizes: Vec<i64> = records.iter().map(|r| r.file_size).collect();
    let created_ats: Vec<i64> = records.iter().map(|r| r.created_at).collect();
    let modified_ats: Vec<i64> = records.iter().map(|r| r.modified_at).collect();

    let schema = create_schema();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            create_null_embedding_array(VISUAL_EMBEDDING_DIM, len),
            Arc::new(StringArray::from(vec![None::<&str>; len])) as ArrayRef,
            create_null_embedding_array(OCR_EMBEDDING_DIM, len),
            Arc::new(StringArray::from(file_types)) as ArrayRef,
            Arc::new(Int64Array::from(file_sizes)) as ArrayRef,
            Arc::new(TimestampMillisecondArray::from(created_ats)) as ArrayRef,
            Arc::new(TimestampMillisecondArray::from(modified_ats)) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;

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
        let escaped = path.replace('\'', "''");
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

pub async fn get_image_by_path(table: &Table, path: &str) -> Result<Option<i64>, String> {
    let escaped = path.replace('\'', "''");
    let batches: Vec<RecordBatch> = table
        .query()
        .only_if(format!("path = '{}'", escaped))
        .select(lancedb::query::Select::Columns(vec![
            "modified_at".to_string(),
        ]))
        .execute()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?
        .try_collect()
        .await
        .map_err(|e: lancedb::Error| e.to_string())?;

    if batches.is_empty() {
        return Ok(None);
    }

    let batch = &batches[0];
    if batch.num_rows() == 0 {
        return Ok(None);
    }

    let modified_col = batch
        .column_by_name("modified_at")
        .ok_or("modified_at column not found")?
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .ok_or("modified_at column is not a timestamp array")?;

    Ok(Some(modified_col.value(0)))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageInfo {
    pub path: String,
    pub file_type: String,
    pub file_size: i64,
    pub created_at: i64,
    pub modified_at: i64,
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
            .as_any().downcast_ref::<Int64Array>().ok_or("file_size not i64")?;
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
