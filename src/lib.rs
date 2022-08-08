use std::env;
use args::Args;
use getopts::Occur;
use thiserror::Error;

mod training_data;
mod interactive_canvas_widget;
mod gui;
mod data;
mod io_ext;
mod train;

static PROGRAM_NAME: &str = "digit_recognition";
static PROGRAM_DESCRIPTION: &str =
    "A simple AI-driven program for classifying pictures based on what digit is written on them";

static KEY_IMAGES_FILE: &str = "IMAGES";
static KEY_LABELS_FILE: &str = "LABELS";

#[derive(Error, Debug)]
pub enum ErrorKind {
    #[error(transparent)]
    GuiError(#[from] gui::ErrorKind),

    #[error("cannot read training dataset")]
    FailedToReadTrainingDataset(#[from] training_data::ErrorKind),

    #[error(transparent)]
    CliError(#[from] args::ArgsError)
}

pub type Result<T> = std::result::Result<T, ErrorKind>;

struct TrainingOption {
    images_file: String,
    labels_file: String
}

enum Action {
    ShowGui,
    Train(TrainingOption)
}

pub fn launch() -> Result<()> {
    let action = parse_args()?;

    match action {
        Action::ShowGui => gui::launch(|_| {})?,
        Action::Train(_) => {}
    }

    Ok(())
}

fn parse_args() -> Result<Action> {
    let mut args = Args::new(PROGRAM_NAME, PROGRAM_DESCRIPTION);

    args.flag("t", "train", "Start training using provided dataset");

    args.option("i",
                "images",
                "File containing images used for training",
                "IMAGES",
                Occur::Optional,
                env::var(KEY_IMAGES_FILE).ok());

    args.option("l",
                "labels",
                "File containing labels used for training",
                "LABELS",
                Occur::Optional,
                env::var(KEY_LABELS_FILE).ok());

    args.parse_from_cli()?;

    return if args.has_value("train") {
        let images_file: String = args.value_of("images")?;
        let labels_file: String = args.value_of("labels")?;

        Ok(Action::Train(TrainingOption { images_file, labels_file }))
    } else {
        Ok(Action::ShowGui)
    }
}
