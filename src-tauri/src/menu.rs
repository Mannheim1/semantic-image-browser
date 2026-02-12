//! Application menu bar construction.

use tauri::menu::{CheckMenuItemBuilder, MenuBuilder, MenuItemBuilder, Menu, PredefinedMenuItem, SubmenuBuilder};
use tauri::Wry;

use crate::config::AppConfig;

pub fn build_menu(app: &mut tauri::App, config: &AppConfig) -> Result<Menu<Wry>, Box<dyn std::error::Error>> {
    let file_menu = SubmenuBuilder::new(app, "&File")
        .item(&MenuItemBuilder::new("&Add Folder...").id("add_folder").accelerator("CmdOrCtrl+O").build(app)?)
        .item(&MenuItemBuilder::new("&Rescan All").id("rescan").accelerator("CmdOrCtrl+R").build(app)?)
        .item(&MenuItemBuilder::new("&Manage Folders...").id("manage_folders").build(app)?)
        .separator()
        .item(&MenuItemBuilder::new("&View Files").id("view_files").build(app)?)
        .separator()
        .item(&MenuItemBuilder::new("Clear &Thumbnails").id("clear_thumbnails").build(app)?)
        .item(&MenuItemBuilder::new("Clear &Database").id("clear_database").build(app)?)
        .separator()
        .item(&PredefinedMenuItem::quit(app, None)?)
        .build()?;

    let edit_menu = SubmenuBuilder::new(app, "&Edit")
        .item(&PredefinedMenuItem::undo(app, None)?)
        .item(&PredefinedMenuItem::redo(app, None)?)
        .separator()
        .item(&PredefinedMenuItem::cut(app, None)?)
        .item(&PredefinedMenuItem::copy(app, None)?)
        .item(&PredefinedMenuItem::paste(app, None)?)
        .item(&PredefinedMenuItem::select_all(app, None)?)
        .build()?;

    let search_menu = SubmenuBuilder::new(app, "&Search")
        .item(&CheckMenuItemBuilder::new("&Lexical OCR").id("ocr_lexical").build(app)?)
        .item(&CheckMenuItemBuilder::new("&Semantic OCR").id("ocr_semantic").build(app)?)
        .separator()
        .item(&SubmenuBuilder::new(app, "Sort &by...")
            .item(&MenuItemBuilder::new("&Relevance").id("sort_relevance").build(app)?)
            .separator()
            .item(&SubmenuBuilder::new(app, "Date &Created")
                .item(&MenuItemBuilder::new("&Ascending").id("sort_created_asc").build(app)?)
                .item(&MenuItemBuilder::new("&Descending").id("sort_created_desc").build(app)?)
                .build()?)
            .item(&SubmenuBuilder::new(app, "Date &Modified")
                .item(&MenuItemBuilder::new("&Ascending").id("sort_modified_asc").build(app)?)
                .item(&MenuItemBuilder::new("&Descending").id("sort_modified_desc").build(app)?)
                .build()?)
            .item(&SubmenuBuilder::new(app, "File &Size")
                .item(&MenuItemBuilder::new("&Ascending").id("sort_size_asc").build(app)?)
                .item(&MenuItemBuilder::new("&Descending").id("sort_size_desc").build(app)?)
                .build()?)
            .build()?)
        .build()?;

    // Determine which runtime is currently active
    let active_runtime = config.runtime_type.as_deref().unwrap_or("cpu");

    let runtime_submenu = SubmenuBuilder::new(app, "Select &Runtime (Restart required)")
        .item(&CheckMenuItemBuilder::new("CPU").id("runtime_cpu").checked(active_runtime == "cpu").build(app)?)
        .item(&CheckMenuItemBuilder::new("GPU (DirectML)").id("runtime_directml").enabled(false).checked(active_runtime == "directml").build(app)?)
        .item(&CheckMenuItemBuilder::new("GPU (CUDA)").id("runtime_cuda").checked(active_runtime == "cuda" || active_runtime == "gpu").build(app)?)
        .build()?;

    let model_menu = SubmenuBuilder::new(app, "&Model")
        .item(&runtime_submenu)
        .separator()
        .item(&MenuItemBuilder::new("&Runtime settings...").id("model_settings").build(app)?)
        .build()?;

    let help_menu = SubmenuBuilder::new(app, "&Help")
        .item(&MenuItemBuilder::new("&About").id("about").build(app)?)
        .item(&MenuItemBuilder::new("View &Controls").id("view_controls").build(app)?)
        .build()?;

    let menu = if config.debug_mode {
        let debug_menu = SubmenuBuilder::new(app, "&Debug")
            .item(&MenuItemBuilder::new("Debug mode enabled").id("debug_mode_enabled").enabled(false).build(app)?)
            .separator()
            .item(&CheckMenuItemBuilder::new("&Benchmarking").id("toggle_benchmarking").checked(config.benchmarking).build(app)?)
            .separator()
            .item(&MenuItemBuilder::new("&Dependency Paths...").id("show_dependency_paths").build(app)?)
            .build()?;
        MenuBuilder::new(app)
            .items(&[&file_menu, &edit_menu, &search_menu, &model_menu, &help_menu, &debug_menu])
            .build()?
    } else {
        MenuBuilder::new(app)
            .items(&[&file_menu, &edit_menu, &search_menu, &model_menu, &help_menu])
            .build()?
    };

    Ok(menu)
}
