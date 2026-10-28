//! Startup identity for the interactive CLI.
//!
//! Product builds expose exactly two visual paths: an embedded Sixel frame for
//! terminals known to support Sixel, and a code-native TINA wordmark everywhere
//! else. VS Code's integrated terminal always uses the TINA path.

use std::fmt::Write as FmtWrite;
use std::io::{self, Write};

const LUNA_SIXEL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/luna_splash.sixel"
));
const LUNA_SIXEL_960: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/luna_splash_960.sixel"
));
const LUNA_SIXEL_1280: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/luna_splash_1280.sixel"
));

const MAX_WRITE_BYTES: usize = 12 * 1024;
const DEFAULT_WIDTH: usize = 80;
const DEFAULT_HEIGHT: usize = 24;
const SIXEL_MIN_WIDTH: usize = 72;
const SIXEL_MIN_HEIGHT: usize = 22;
const TINA_LOGO_MASK: [&str; 5] = [
    "#####  ###  #   #   ### ",
    "  #     #   ##  #  #   #",
    "  #     #   # # #  #####",
    "  #     #   #  ##  #   #",
    "  #    ###  #   #  #   #",
];
const TINA_GRADIENT: [Rgb; 5] = [
    Rgb(75, 103, 180),
    Rgb(73, 151, 220),
    Rgb(121, 199, 235),
    Rgb(207, 229, 252),
    Rgb(160, 148, 224),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rgb(u8, u8, u8);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SplashStyle {
    #[default]
    Auto,
    Tina,
    Sixel,
}

impl SplashStyle {
    fn from_environment() -> Self {
        std::env::var("CQ_SPLASH_STYLE")
            .ok()
            .map(|value| Self::from_value(&value))
            .unwrap_or_default()
    }

    fn from_value(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "tina" => Self::Tina,
            "sixel" => Self::Sixel,
            _ => Self::Auto,
        }
    }
}

pub(crate) fn show() -> io::Result<()> {
    show_terminal(SplashStyle::from_environment())
}

fn show_terminal(style: SplashStyle) -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    ensure_utf8_console();

    let (width, height) = terminal_size();
    let no_color = std::env::var_os("NO_COLOR").is_some();
    if !no_color && should_render_sixel(style, width, height) {
        return write_sixel_frame(&mut stderr, sixel_payload_for_terminal(width, height));
    }

    let body = render_tina_wordmark(width, height, !no_color);
    write_terminal_frame(&mut stderr, &body)
}

fn should_render_sixel(style: SplashStyle, width: usize, height: usize) -> bool {
    let term_program = std::env::var("TERM_PROGRAM").ok();
    let term = std::env::var("TERM").ok();
    let vscode_injection = std::env::var("VSCODE_INJECTION").ok();
    should_render_sixel_with_hints(
        style,
        width,
        height,
        vscode_terminal_hint(term_program.as_deref(), vscode_injection.as_deref()),
        std::env::var_os("WT_SESSION").is_some(),
        term_program.as_deref(),
        term.as_deref(),
    )
}

fn should_render_sixel_with_hints(
    style: SplashStyle,
    width: usize,
    height: usize,
    vscode_terminal: bool,
    windows_terminal_session: bool,
    term_program: Option<&str>,
    term: Option<&str>,
) -> bool {
    if vscode_terminal || width < SIXEL_MIN_WIDTH || height < SIXEL_MIN_HEIGHT {
        return false;
    }

    match style {
        SplashStyle::Tina => false,
        SplashStyle::Sixel => true,
        SplashStyle::Auto => sixel_terminal_hint(windows_terminal_session, term_program, term),
    }
}

fn vscode_terminal_hint(term_program: Option<&str>, vscode_injection: Option<&str>) -> bool {
    term_program.is_some_and(|value| value.eq_ignore_ascii_case("vscode"))
        || vscode_injection.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn sixel_terminal_hint(
    windows_terminal_session: bool,
    term_program: Option<&str>,
    term: Option<&str>,
) -> bool {
    if windows_terminal_session {
        return true;
    }
    if term.is_some_and(|value| value.to_ascii_lowercase().contains("sixel")) {
        return true;
    }
    term_program
        .map(str::to_ascii_lowercase)
        .is_some_and(|program| {
            matches!(
                program.as_str(),
                "contour" | "foot" | "konsole" | "mintty" | "mlterm" | "rio" | "wezterm"
            )
        })
}

fn sixel_payload_for_terminal(width: usize, height: usize) -> &'static str {
    if width >= 160 && height >= 45 {
        LUNA_SIXEL_1280
    } else if width >= 120 && height >= 34 {
        LUNA_SIXEL_960
    } else {
        LUNA_SIXEL
    }
}

fn render_tina_wordmark(width: usize, height: usize, color_enabled: bool) -> String {
    let frame_width = width.saturating_sub(1).max(1);
    let mask_width = TINA_LOGO_MASK[0].chars().count();
    let horizontal_scale = usize::from(frame_width >= mask_width * 2 + 4) + 1;
    let logo_width = mask_width * horizontal_scale;
    if logo_width > frame_width || height < TINA_LOGO_MASK.len() + 4 {
        return render_compact_tina(width, height, color_enabled);
    }

    let content_rows = TINA_LOGO_MASK.len() + 2;
    let available_rows = height.saturating_sub(2);
    let top_padding = available_rows.saturating_sub(content_rows).min(6) / 2;
    let left_padding = frame_width.saturating_sub(logo_width) / 2;
    let mut body = String::with_capacity(logo_width.saturating_mul(64));
    for _ in 0..top_padding {
        body.push('\n');
    }

    for (row_index, row) in TINA_LOGO_MASK.iter().enumerate() {
        for _ in 0..left_padding {
            body.push(' ');
        }
        for (mask_column, cell) in row.chars().enumerate() {
            for sub_column in 0..horizontal_scale {
                if cell != '#' {
                    body.push(' ');
                    continue;
                }
                if color_enabled {
                    let column = mask_column * horizontal_scale + sub_column;
                    let color =
                        tina_gradient_color(column, row_index, logo_width, TINA_LOGO_MASK.len());
                    let _ = write!(body, "\x1b[38;2;{};{};{}m█", color.0, color.1, color.2);
                } else {
                    body.push('#');
                }
            }
        }
        if color_enabled {
            body.push_str("\x1b[0m");
        }
        body.push('\n');
    }

    body.push('\n');
    push_centered_line(
        &mut body,
        frame_width,
        "Tidal Luna / Tina Engine",
        color_enabled.then_some(Rgb(194, 220, 247)),
    );
    body
}

fn render_compact_tina(width: usize, height: usize, color_enabled: bool) -> String {
    let frame_width = width.saturating_sub(1).max(1);
    let mut body = String::with_capacity(256);
    if height >= 6 {
        body.push('\n');
    }
    push_centered_gradient_line(&mut body, frame_width, "T I N A", color_enabled);
    if height >= 5 {
        push_centered_line(
            &mut body,
            frame_width,
            "Tidal Luna",
            color_enabled.then_some(Rgb(151, 191, 235)),
        );
    }
    body
}

fn tina_gradient_color(column: usize, row: usize, width: usize, height: usize) -> Rgb {
    let vertical_weight = 2;
    let diagonal = column.saturating_add(
        height
            .saturating_sub(1)
            .saturating_sub(row)
            .saturating_mul(vertical_weight),
    );
    let span = width
        .saturating_sub(1)
        .saturating_add(height.saturating_sub(1).saturating_mul(vertical_weight))
        .max(1);
    let position = diagonal as f32 / span as f32;
    let scaled = position * (TINA_GRADIENT.len() - 1) as f32;
    let first = (scaled.floor() as usize).min(TINA_GRADIENT.len() - 2);
    let fraction = scaled - first as f32;
    lerp_rgb(TINA_GRADIENT[first], TINA_GRADIENT[first + 1], fraction)
}

fn lerp_rgb(start: Rgb, end: Rgb, amount: f32) -> Rgb {
    let channel = |from: u8, to: u8| {
        (from as f32 + (to as f32 - from as f32) * amount)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Rgb(
        channel(start.0, end.0),
        channel(start.1, end.1),
        channel(start.2, end.2),
    )
}

fn push_centered_line(body: &mut String, width: usize, text: &str, color: Option<Rgb>) {
    let text_width = text.chars().count().min(width);
    let left_padding = width.saturating_sub(text_width) / 2;
    for _ in 0..left_padding {
        body.push(' ');
    }
    if let Some(color) = color {
        let _ = write!(body, "\x1b[38;2;{};{};{}m", color.0, color.1, color.2);
    }
    body.extend(text.chars().take(text_width));
    if color.is_some() {
        body.push_str("\x1b[0m");
    }
    body.push('\n');
}

fn push_centered_gradient_line(body: &mut String, width: usize, text: &str, color_enabled: bool) {
    let text_width = text.chars().count().min(width);
    let left_padding = width.saturating_sub(text_width) / 2;
    for _ in 0..left_padding {
        body.push(' ');
    }
    for (column, glyph) in text.chars().take(text_width).enumerate() {
        if !color_enabled || glyph == ' ' {
            body.push(glyph);
            continue;
        }
        let color = tina_gradient_color(column, 0, text_width.max(1), 1);
        let _ = write!(
            body,
            "\x1b[1;38;2;{};{};{}m{}",
            color.0, color.1, color.2, glyph
        );
    }
    if color_enabled {
        body.push_str("\x1b[0m");
    }
    body.push('\n');
}

fn write_sixel_frame<W: Write>(out: &mut W, payload: &str) -> io::Result<()> {
    let prefix = b"\n\x1b[?2026h\x1b[?7l\x1b[?25l";
    let mut first_error = out.write_all(prefix).err();
    if first_error.is_none() {
        first_error = write_chunked(out, payload).err();
    }
    if let Err(error) = out.write_all(b"\x1b[0m\n\x1b[?7h\x1b[?25h\x1b[?2026l\n") {
        if first_error.is_none() {
            first_error = Some(error);
        }
    }
    if let Err(error) = out.flush() {
        if first_error.is_none() {
            first_error = Some(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn write_terminal_frame<W: Write>(out: &mut W, body: &str) -> io::Result<()> {
    let prefix = b"\n\x1b[?2026h\x1b[?7l\x1b[?25l";
    let mut first_error = out.write_all(prefix).err();
    if first_error.is_none() {
        first_error = write_chunked(out, body).err();
    }
    if let Err(error) = out.write_all(b"\x1b[0m\x1b[?7h\x1b[?25h\x1b[?2026l\n") {
        if first_error.is_none() {
            first_error = Some(error);
        }
    }
    if let Err(error) = out.flush() {
        if first_error.is_none() {
            first_error = Some(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn write_chunked<W: Write>(out: &mut W, body: &str) -> io::Result<()> {
    for line in body.split_inclusive('\n') {
        if line.len() <= MAX_WRITE_BYTES {
            out.write_all(line.as_bytes())?;
            continue;
        }

        let mut chunk = String::new();
        for glyph in line.chars() {
            if !chunk.is_empty() && chunk.len() + glyph.len_utf8() > MAX_WRITE_BYTES {
                out.write_all(chunk.as_bytes())?;
                chunk.clear();
            }
            chunk.push(glyph);
        }
        if !chunk.is_empty() {
            out.write_all(chunk.as_bytes())?;
        }
    }
    Ok(())
}

fn terminal_size() -> (usize, usize) {
    let env_width = env_dimension("COLUMNS");
    let env_height = env_dimension("LINES");
    let native = native_terminal_size();
    let width = env_width
        .or(native.map(|size| size.0))
        .unwrap_or(DEFAULT_WIDTH);
    let height = env_height
        .or(native.map(|size| size.1))
        .unwrap_or(DEFAULT_HEIGHT);
    normalise_size(width, height)
}

fn env_dimension(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn normalise_size(width: usize, height: usize) -> (usize, usize) {
    (width.clamp(40, 240), height.clamp(12, 80))
}

#[cfg(not(windows))]
fn native_terminal_size() -> Option<(usize, usize)> {
    None
}

#[cfg(windows)]
fn native_terminal_size() -> Option<(usize, usize)> {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Coord {
        x: i16,
        y: i16,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SmallRect {
        left: i16,
        top: i16,
        right: i16,
        bottom: i16,
    }
    #[repr(C)]
    struct ConsoleScreenBufferInfo {
        size: Coord,
        cursor: Coord,
        attributes: u16,
        window: SmallRect,
        max_window_size: Coord,
    }

    // SAFETY: these declarations match the Kernel32 console ABI. The caller
    // below validates the returned handle and only passes a live pointer to
    // the initialized output structure.
    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(standard_handle: i32) -> *mut c_void;
        fn GetConsoleScreenBufferInfo(
            console_output: *mut c_void,
            info: *mut ConsoleScreenBufferInfo,
        ) -> i32;
    }

    const STD_ERROR_HANDLE: i32 = -12;
    let mut info = MaybeUninit::<ConsoleScreenBufferInfo>::uninit();
    // SAFETY: `GetStdHandle` takes a constant selector and returns an opaque
    // process-owned handle; no Rust reference is formed from the raw value.
    let handle = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
    if handle.is_null() || handle as isize == -1 {
        return None;
    }
    // SAFETY: `info` points to writable MaybeUninit storage with the exact C
    // layout expected by `GetConsoleScreenBufferInfo`; the handle was checked
    // above and the function reports success before initialization is read.
    let initialized = unsafe { GetConsoleScreenBufferInfo(handle, info.as_mut_ptr()) };
    if initialized == 0 {
        return None;
    }
    // SAFETY: the preceding successful call initialized every field of `info`.
    let size = unsafe { info.assume_init() };
    let width = size
        .window
        .right
        .saturating_sub(size.window.left)
        .saturating_add(1);
    let height = size
        .window
        .bottom
        .saturating_sub(size.window.top)
        .saturating_add(1);
    if width > 0 && height > 0 {
        Some((width as usize, height as usize))
    } else {
        None
    }
}

#[cfg(windows)]
fn ensure_utf8_console() {
    use std::ffi::c_void;
    // SAFETY: these declarations match the Kernel32 console ABI. All pointer
    // arguments below refer to local scalar storage or an opaque handle that
    // is checked before use.
    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn GetConsoleMode(console_handle: *mut c_void, mode: *mut u32) -> i32;
        fn GetStdHandle(standard_handle: i32) -> *mut c_void;
        fn SetConsoleMode(console_handle: *mut c_void, mode: u32) -> i32;
        fn SetConsoleOutputCP(code_page: u32) -> i32;
    }
    const STD_ERROR_HANDLE: i32 = -12;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    let mut mode = 0u32;
    // SAFETY: the handle is validated before it is passed to the console
    // functions; `mode` is valid writable storage for the duration of calls.
    let handle = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
    if !handle.is_null() && handle as isize != -1 {
        // SAFETY: `handle` is a live console handle returned for this process,
        // and `mode` is writable storage with the ABI-required alignment.
        let console_mode_available = unsafe { GetConsoleMode(handle, &mut mode) } != 0;
        if !console_mode_available {
            return;
        }
        // SAFETY: the validated console handle belongs to this process and
        // the code page/mode values are documented Kernel32 inputs.
        let _ = unsafe { SetConsoleOutputCP(65001) };
        // SAFETY: `handle` passed the mode query above and remains valid for
        // this synchronous mode update; the bitmask is a documented flag.
        let _ = unsafe { SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) };
    }
}

#[cfg(not(windows))]
fn ensure_utf8_console() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(line: &str) -> String {
        let mut plain = String::new();
        let mut escape = false;
        for glyph in line.chars() {
            if escape {
                if glyph.is_ascii_alphabetic() {
                    escape = false;
                }
            } else if glyph == '\u{1b}' {
                escape = true;
            } else {
                plain.push(glyph);
            }
        }
        plain
    }

    fn visible_width(line: &str) -> usize {
        strip_ansi(line).chars().count()
    }

    #[test]
    fn only_auto_tina_and_sixel_are_selectable() {
        assert_eq!(SplashStyle::from_value("auto"), SplashStyle::Auto);
        assert_eq!(SplashStyle::from_value("tina"), SplashStyle::Tina);
        assert_eq!(SplashStyle::from_value("sixel"), SplashStyle::Sixel);
        assert_eq!(SplashStyle::from_value("pet"), SplashStyle::Auto);
        assert_eq!(SplashStyle::from_value("bitmap"), SplashStyle::Auto);
    }

    #[test]
    fn tina_wordmark_uses_brand_colors_and_fits() {
        let body = render_tina_wordmark(80, 24, true);
        assert!(body.contains('█'));
        assert!(body.contains("\x1b[38;2;"));
        assert!(strip_ansi(&body).contains("Tidal Luna / Tina Engine"));
        for line in body.lines() {
            assert!(visible_width(line) <= 79, "line is too wide: {line:?}");
        }
    }

    #[test]
    fn no_color_tina_is_control_sequence_free() {
        let body = render_tina_wordmark(80, 24, false);
        assert!(body.contains('#'));
        assert!(body.contains("Tidal Luna / Tina Engine"));
        assert!(!body.contains('\u{1b}'));
    }

    #[test]
    fn compact_tina_is_small_and_readable() {
        let body = render_compact_tina(30, 8, true);
        assert!(strip_ansi(&body).contains("T I N A"));
        assert!(strip_ansi(&body).contains("Tidal Luna"));
        for line in body.lines() {
            assert!(visible_width(line) <= 29);
        }
    }

    #[test]
    fn tina_wordmark_mask_and_gradient_are_stable() {
        let width = TINA_LOGO_MASK[0].chars().count();
        assert!(TINA_LOGO_MASK
            .iter()
            .all(|line| line.chars().count() == width));
        assert_eq!(
            tina_gradient_color(0, TINA_LOGO_MASK.len() - 1, width, TINA_LOGO_MASK.len()),
            TINA_GRADIENT[0]
        );
        assert_eq!(
            tina_gradient_color(width - 1, 0, width, TINA_LOGO_MASK.len()),
            TINA_GRADIENT[TINA_GRADIENT.len() - 1]
        );
    }

    #[test]
    fn normalise_size_rejects_extreme_values() {
        assert_eq!(normalise_size(1, 1), (40, 12));
        assert_eq!(normalise_size(999, 999), (240, 80));
    }

    #[test]
    fn sixel_assets_are_palette_encoded_and_terminated() {
        for (payload, header, minimum_len) in [
            (LUNA_SIXEL, "\u{1b}Pq\"1;1;640;360", 100_000),
            (LUNA_SIXEL_960, "\u{1b}Pq\"1;1;960;540", 200_000),
            (LUNA_SIXEL_1280, "\u{1b}Pq\"1;1;1280;720", 300_000),
        ] {
            assert!(payload.starts_with(header));
            assert!(payload.ends_with("\u{1b}\\"));
            assert!(payload.contains("#255;2;"));
            assert!(payload.len() > minimum_len);
        }
    }

    #[test]
    fn sixel_resolution_tracks_terminal_size() {
        assert_eq!(sixel_payload_for_terminal(80, 24), LUNA_SIXEL);
        assert_eq!(sixel_payload_for_terminal(119, 40), LUNA_SIXEL);
        assert_eq!(sixel_payload_for_terminal(120, 34), LUNA_SIXEL_960);
        assert_eq!(sixel_payload_for_terminal(159, 44), LUNA_SIXEL_960);
        assert_eq!(sixel_payload_for_terminal(160, 45), LUNA_SIXEL_1280);
        assert_eq!(sixel_payload_for_terminal(240, 80), LUNA_SIXEL_1280);
    }

    #[test]
    fn vscode_always_selects_tina() {
        assert!(vscode_terminal_hint(Some("vscode"), None));
        assert!(vscode_terminal_hint(None, Some("1")));
        assert!(!should_render_sixel_with_hints(
            SplashStyle::Sixel,
            160,
            45,
            true,
            true,
            Some("WezTerm"),
            Some("xterm-sixel"),
        ));
    }

    #[test]
    fn auto_sixel_selection_is_conservative() {
        assert!(should_render_sixel_with_hints(
            SplashStyle::Auto,
            80,
            24,
            false,
            true,
            None,
            None,
        ));
        assert!(should_render_sixel_with_hints(
            SplashStyle::Auto,
            80,
            24,
            false,
            false,
            Some("WezTerm"),
            None,
        ));
        assert!(should_render_sixel_with_hints(
            SplashStyle::Auto,
            80,
            24,
            false,
            false,
            None,
            Some("xterm-sixel"),
        ));
        assert!(!should_render_sixel_with_hints(
            SplashStyle::Auto,
            80,
            24,
            false,
            false,
            None,
            Some("xterm-256color"),
        ));
        assert!(!should_render_sixel_with_hints(
            SplashStyle::Tina,
            160,
            45,
            false,
            true,
            Some("WezTerm"),
            Some("xterm-sixel"),
        ));
        assert!(!should_render_sixel_with_hints(
            SplashStyle::Sixel,
            60,
            20,
            false,
            true,
            None,
            None,
        ));
    }
}
