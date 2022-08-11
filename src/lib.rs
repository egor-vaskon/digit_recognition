extern crate core;

use std::{env, io};
use std::fs::File;
use args::Args;
use getopts::Occur;
use nalgebra::DVector;
use thiserror::Error;
use crate::network::{INPUT_LAYER_SIZE, NeuralNetwork};
use crate::training_data::TrainingDataset;

mod training_data;
mod interactive_canvas_widget;
mod gui;
mod data;
mod io_ext;
mod network;

static PROGRAM_NAME: &str = "digit_recognition";
static PROGRAM_DESCRIPTION: &str =
    "A simple AI-driven program for classifying pictures based on what digit is written on them";

static KEY_IMAGES_FILE: &str = "IMAGES";
static KEY_LABELS_FILE: &str = "LABELS";

#[derive(Error, Debug)]
pub enum ErrorKind {
    #[error(transparent)]
    GuiError(#[from] gui::ErrorKind),

    #[error(transparent)]
    NeuralNetworkError(#[from] network::ErrorKind),

    #[error("cannot read training dataset ({0})")]
    FailedToReadTrainingDataset(#[from] training_data::ErrorKind),

    #[error("cannot read training dataset ({0})")]
    CannotReadTrainingDataset(#[from] io::Error),

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
    let mut neural_network =
        NeuralNetwork::load("neural_network.json")
            .unwrap_or(NeuralNetwork::new_untrained());

    match action {
        Action::ShowGui => gui::launch(move |img_loader| {
            let image = img_loader.load_image(28).unwrap();
            let image_pixels = image.pixels();
            let input =
                DVector::from_iterator(28*28, image_pixels.iter()
                    .map(|x| 1.0 - ((*x as f64) / 255.0) - 0.5));

            let output = neural_network.compute(input);
            let (digit, chance) = output
                .as_slice()
                .iter()
                .enumerate()
                .fold((0u8, f64::NEG_INFINITY), |(acc_i, acc_v), (i, x)| {
                    let (i, x) = (i as u8, *x);
                    return if x > acc_v {
                        (i, x)
                    } else {
                        (acc_i, acc_v)
                    }
                });

            (digit, chance)
        })?,
        Action::Train(opts) => {
            let images = File::open(opts.images_file)?;
            let labels = File::open(opts.labels_file)?;

            let dataset =
                TrainingDataset::from_readers(images, labels)?;

            let dataset_size = dataset.size();

            let mut count = vec![0; 10];

            for (i, example) in dataset.take(10_000).enumerate() {
                match example {
                    Ok(example) => {
                        let pixels =
                            DVector::from_iterator(28*28, example
                                .image()
                                .pixels()
                                .iter()
                                .map(|px| ((*px as f64) / 255.0) - 0.5));

                        println!("input: {}", pixels.mean());

                        let mut expected_output = DVector::zeros(10);
                        expected_output[example.label().digit() as usize] = 1.0;

                        neural_network.train(pixels, &expected_output);

                        let completion = ((i+1) as f64) / (dataset_size as f64);
                        println!("Finished {} training examples ({:.2}%)", i+1, completion*100.0);

                        count[example.label().digit() as usize] += 1;
                    },
                    Err(_) => break
                }
            }

            println!("cc: {}", DVector::from_column_slice(&count));
            neural_network.save("neural_network.json");
        }
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

    return if args.value_of::<bool>("train")? {
        let images_file: String = args.value_of("images")?;
        let labels_file: String = args.value_of("labels")?;

        Ok(Action::Train(TrainingOption { images_file, labels_file }))
    } else {
        Ok(Action::ShowGui)
    }
}
