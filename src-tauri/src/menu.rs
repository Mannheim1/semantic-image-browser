//! Application menu bar construction.

use tauri::menu::{CheckMenuItemBuilder, MenuBuilder, MenuItemBuilder, Menu, PredefinedMenuItem, SubmenuBuilder};
use tauri::Wry;

use crate::config::AppConfig;

pub fn build_menu(app: &mut tauri::App, config: &AppConfig) -> Result<Menu<Wry>, Box<dyn std::error::Error>> {
    // TODO: Give frequently-used menu buttons functioning commands.
    // ── macOS app menu ──────────────────────────────────────────────────────
    #[cfg(target_os = "macos")]
    let app_menu = SubmenuBuilder::new(app, app.package_info().name.as_str())
        .item(&MenuItemBuilder::new("About").id("about").build(app)?)
        .separator()
        .item(&PredefinedMenuItem::hide(app, None)?)
        .item(&PredefinedMenuItem::hide_others(app, None)?)
        .item(&PredefinedMenuItem::show_all(app, None)?)
        .separator()
        .item(&PredefinedMenuItem::quit(app, None)?)
        .build()?;

    // ── File menu ───────────────────────────────────────────────────────────
    let file_menu = {
        let b = SubmenuBuilder::new(app, "&File")
            .item(&MenuItemBuilder::new("&Add Folder...").id("add_folder").build(app)?)
            .item(&MenuItemBuilder::new("&Rescan All").id("rescan").build(app)?)
            .item(&MenuItemBuilder::new("&Manage Folders...").id("manage_folders").build(app)?)
            .separator()
            .item(&MenuItemBuilder::new("&Open App Data Folder").id("view_files").build(app)?)
            .separator()
            .item(&MenuItemBuilder::new("Clear &Thumbnails").id("clear_thumbnails").build(app)?)
            .item(&MenuItemBuilder::new("Clear &Database").id("clear_database").build(app)?);

        #[cfg(not(target_os = "macos"))]
        let b = b.separator().item(&PredefinedMenuItem::quit(app, None)?);

        b.build()?
    };

    // ── View menu ───────────────────────────────────────────────────────────
    #[cfg(target_os = "macos")]
    let toggle_fullscreen_label = "Toggle Fullscreen";
    #[cfg(not(target_os = "macos"))]
    let toggle_fullscreen_label = "Toggle &Fullscreen";

    let _view_menu = SubmenuBuilder::new(app, "&View")
        .item(&MenuItemBuilder::new("Zoom &In").id("zoom_in").accelerator("CmdOrCtrl+=").build(app)?)
        .item(&MenuItemBuilder::new("Zoom &Out").id("zoom_out").accelerator("CmdOrCtrl+-").build(app)?)
        .item(&MenuItemBuilder::new("&Reset Zoom").id("reset_zoom").accelerator("CmdOrCtrl+0").build(app)?)
        .separator()
        .item(&MenuItemBuilder::new(toggle_fullscreen_label).id("toggle_fullscreen").accelerator("F11").build(app)?)
        .build()?;

    // ── Search menu ─────────────────────────────────────────────────────────
    #[cfg(target_os = "macos")]
    let lexical_ocr_label = "Lexical OCR";
    #[cfg(not(target_os = "macos"))]
    let lexical_ocr_label = "&Lexical OCR";

    #[cfg(target_os = "macos")]
    let semantic_ocr_label = "Semantic OCR";
    #[cfg(not(target_os = "macos"))]
    let semantic_ocr_label = "&Semantic OCR";

    // TODO: Ungrey/enable OCR toggles when the spec sheet's designated stage is reached.
    let search_menu = SubmenuBuilder::new(app, "&Search")
        .item(&CheckMenuItemBuilder::new(lexical_ocr_label).id("ocr_lexical").checked(false).enabled(false).build(app)?)
        .item(&CheckMenuItemBuilder::new(semantic_ocr_label).id("ocr_semantic").checked(false).enabled(false).build(app)?)
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

    // ── Help menu ───────────────────────────────────────────────────────────
    let help_menu = {
        #[cfg(not(target_os = "macos"))]
        let b = SubmenuBuilder::new(app, "&Help")
            .item(&MenuItemBuilder::new("&About").id("about").build(app)?);

        #[cfg(target_os = "macos")]
        let b = SubmenuBuilder::new(app, "&Help");

        b.item(&MenuItemBuilder::new("View &Controls").id("view_controls").build(app)?)
            .build()?
    };

    // ── Debug menu ──────────────────────────────────────────────────────────
    let menu = if config.debug_mode {
        #[cfg(target_os = "macos")]
        let benchmarking_label = "Benchmarking";
        #[cfg(not(target_os = "macos"))]
        let benchmarking_label = "&Benchmarking";

        let debug_menu = SubmenuBuilder::new(app, "&Debug")
            .item(&MenuItemBuilder::new("Debug mode enabled").id("debug_mode_enabled").enabled(false).build(app)?)
            .separator()
            .item(&CheckMenuItemBuilder::new(benchmarking_label).id("toggle_benchmarking").checked(false).build(app)?)
            .separator()
            .item(&MenuItemBuilder::new("&Dependency Paths").id("show_dependency_paths").build(app)?)
            .item(&MenuItemBuilder::new("&Test Bundle URLs").id("test_bundle_urls").build(app)?)
            .build()?;

        #[cfg(target_os = "macos")]
        let m = MenuBuilder::new(app)
            // TODO: Re-add the View menu and give its buttons real functionality eventually.
            .items(&[&app_menu, &file_menu, &search_menu, &help_menu, &debug_menu])
            .build()?;

        #[cfg(not(target_os = "macos"))]
        let m = MenuBuilder::new(app)
            // TODO: Re-add the View menu and give its buttons real functionality eventually.
            .items(&[&file_menu, &search_menu, &help_menu, &debug_menu])
            .build()?;

        m
    } else {
        #[cfg(target_os = "macos")]
        let m = MenuBuilder::new(app)
            // TODO: Re-add the View menu and give its buttons real functionality eventually.
            .items(&[&app_menu, &file_menu, &search_menu, &help_menu])
            .build()?;

        #[cfg(not(target_os = "macos"))]
        let m = MenuBuilder::new(app)
            // TODO: Re-add the View menu and give its buttons real functionality eventually.
            .items(&[&file_menu, &search_menu, &help_menu])
            .build()?;

        m
    };

    Ok(menu)
}
