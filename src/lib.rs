use druid::PlatformError;
use thiserror::Error;

mod training;

#[derive(Error, Debug)]
enum ErrorKind {
    #[error("cannot render gui, because a druid error occurred")]
    DruidError(#[from] PlatformError),

    #[error("cannot read training dataset")]
    FailedToReadTrainingDataset(#[from] training::ErrorKind)
}

