use druid::{Data, Lens, AppLauncher, Color, Insets, LocalizedString, MenuDesc, PlatformError, Size, TextAlignment, Widget, WidgetExt, WindowDesc, piet};
use druid::widget::{Button, Flex, FlexParams, Label, SizedBox};
use thiserror::Error;
use crate::data::{Image, ImageSize};
use crate::interactive_canvas_widget::{InteractiveCanvas, InteractiveCanvasState};

#[derive(Error, Debug)]
pub enum ErrorKind {
    #[error("cannot render gui, because of druid error")]
    DruidPlatformError(#[from] PlatformError),

    #[error("cannot copy pixels from image ({0})")]
    CannotCopyPixels(#[from] piet::Error)
}

pub type Result<T> = std::result::Result<T, ErrorKind>;

#[derive(Data, Lens, Clone)]
struct AppState {
    digit: u8,
    accuracy: f64,
    canvas_state: InteractiveCanvasState
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            digit: 9,
            accuracy: 0.99,
            canvas_state: InteractiveCanvasState::builder()
                .with_background(Color::WHITE)
                .with_stroke_brush(Color::BLACK)
                .with_stroke_width(0.036)
                .build()
        }
    }
}

pub struct ImageLoader<'a> {
    canvas: &'a mut InteractiveCanvasState
}

impl ImageLoader<'_> {
    pub fn load_image(&self, size_dimension: u32) -> Result<Image> {
        let pixels = self.canvas.copy_pixels_grayscale(size_dimension)?;
        Ok(Image::builder()
            .with_size(ImageSize::square(size_dimension))
            .with_pixels_row_major(pixels)
            .build())
    }
}

pub fn launch<F>(on_submit: F) -> Result<()>
    where F: Fn(ImageLoader) -> (u8, f64) + 'static
{
    open_window(move |state| {
        let image_loader = ImageLoader { canvas: &mut state.canvas_state };
        (state.digit, state.accuracy) = on_submit(image_loader);
        state.canvas_state.clear();
    })
}

fn open_window<F>(on_submit: F) -> Result<()>
    where F: Fn(&mut AppState) + 'static
{
    let window_menu = MenuDesc::new(LocalizedString::new("window_title"));
    let window = WindowDesc::new(move || build_ui(on_submit))
        .window_size(Size::new(800.0, 600.0))
        .resizable(true)
        .menu(window_menu);

    AppLauncher::with_window(window)
        .use_simple_logger()
        .launch(AppState::default())?;

    Ok(())
}

fn build_ui<F>(on_submit: F) -> impl Widget<AppState>
    where F: Fn(&mut AppState) + 'static
{
    let canvas = InteractiveCanvas::default()
        .with_state(
            |state: &AppState| state.canvas_state.clone());

    let recognized_digit_label = Label::dynamic(|digit, _| format!("{}", digit))
        .with_text_size(60.0)
        .with_text_alignment(TextAlignment::Center)
        .padding(Insets::uniform_xy(10.0, 0.0))
        .background(Color::BLUE)
        .rounded(60.0)
        .lens(AppState::digit);

    let recognized_digit_accuracy_text_label =
        Label::dynamic(|accuracy, _| format!("{:.0}%", accuracy*100.0))
            .with_text_size(33.0)
            .with_text_alignment(TextAlignment::Center)
            .lens(AppState::accuracy);

    let submit_button_label: Label<AppState> =
        Label::new("Resubmit")
            .with_text_size(30.0);

    let submit_button =
        Button::from_label(submit_button_label)
            .on_click(move |_, state, _|
                on_submit(state));

    let controls =
        Flex::column()
            .with_child(Label::new("Result").with_text_size(50.0))
            .with_spacer(20.0)
            .with_child(recognized_digit_label)
            .with_spacer(20.0)
            .with_child(Label::new("chance:").with_text_size(33.0))
            .with_spacer(10.0)
            .with_child(recognized_digit_accuracy_text_label)
            .with_spacer(50.0)
            .with_child(submit_button);

    Flex::row()
        .with_flex_child(canvas, FlexParams::from(1.0))
        .with_child(SizedBox::new(controls).width(200.0))
}