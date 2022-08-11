use std::fmt::{Debug, Display, Formatter};
use std::io;
use std::io::Read;
use typed_io::TypedRead;
use thiserror::Error;
use crate::data::{Image, ImageSize};
use crate::io_ext::{IntoDataIter, ReadData, ReadFromBytes, SimpleDataIter};

const IMAGES_MAGIC: u32 = 0x00000803;
const LABELS_MAGIC: u32 = 0x00000801;

#[derive(Debug, Copy, Clone)]
pub enum DataKind {
    Image,
    Label
}

impl Display for DataKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return match self {
            DataKind::Image => f.write_str("images dataset"),
            DataKind::Label => f.write_str("labels dataset")
        }
    }
}

#[derive(Error, Debug)]
pub enum ErrorKind {
    #[error("expected magic number ({magic:#x}), found {found:#x} (in {dataset_kind})")]
    MagicNotFound {
        found: u32,
        magic: u32,
        dataset_kind: DataKind
    },

    #[error("cannot read dataset due to an I/O error")]
    IO(#[from] io::Error),

    #[error("the number of labels ({label_count}) must match the number of training samples ({sample_count})")]
    InvalidLabelCount {
        sample_count: u32,
        label_count: u32
    }
}

pub type Result<T> = std::result::Result<T, ErrorKind>;

impl ReadFromBytes for ImageSize {
    type Error = io::Error;
    type Config = ();

    fn read_from_bytes(input: &mut impl Read,
                       _config: &Self::Config) -> std::result::Result<Self, Self::Error>
        where Self: Sized
    {
        let width: u32 = input.read_be()?;
        let height: u32 = input.read_be()?;

        Ok(ImageSize {
            width,
            height
        })
    }
}

impl ReadFromBytes for Image {
    type Error = io::Error;
    type Config = ImageSize;

    fn read_from_bytes(input: &mut impl Read,
                       config: &Self::Config) -> std::result::Result<Self, Self::Error>
        where Self: Sized
    {
        let mut pixels = vec![0; config.area()];
        input.read_exact(&mut pixels)?;

        Ok(Image::builder()
            .with_size(*config)
            .with_pixels_row_major(pixels)
            .build())
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Label {
    digit: u8
}

impl Label {
    pub fn new(digit: u8) -> Self {
        Label { digit }
    }

    pub fn digit(&self) -> u8 {
        self.digit
    }
}

impl ReadFromBytes for Label {
    type Error = io::Error;
    type Config = ();

    fn read_from_bytes(input: &mut impl Read,
                       _config: &Self::Config) -> std::result::Result<Self, Self::Error>
        where Self: Sized
    {
        let digit: u8 = input.read_ne()?;
        Ok(Label::new(digit))
    }
}

#[derive(Debug, PartialEq)]
pub struct LabeledTrainingData {
    image: Image,
    label: Label
}

impl LabeledTrainingData {
    pub fn new(image: Image, label: Label) -> LabeledTrainingData {
        LabeledTrainingData {
            image, label
        }
    }

    pub fn image(&self) -> &Image {
        &self.image
    }

    pub fn label(&self) -> &Label {
        &self.label
    }
}

const fn magic(data_kind: DataKind) -> u32 {
    match data_kind {
        DataKind::Image => IMAGES_MAGIC,
        DataKind::Label => LABELS_MAGIC
    }
}

fn verify_magic<R: Read>(input: &mut R, data_kind: DataKind) -> Result<()> {
    let found = input.read_be::<u32>()?;
    return if found == magic(data_kind) {
        Ok(())
    } else {
        Err(ErrorKind::MagicNotFound {
            found,
            magic: magic(data_kind),
            dataset_kind: data_kind
        })
    }
}

struct TrainingImageSet<R: Read> {
    images: SimpleDataIter<Image, R>,
    image_count: u32
}

impl<R: Read> TrainingImageSet<R> {
    fn try_from(mut input: R) -> Result<Self> {
        verify_magic(&mut input, DataKind::Image)?;

        let image_count: u32 = input.read_be()?;
        let image_size: ImageSize = input.read_data(&())?;

        Ok(TrainingImageSet {
            images: input.data_iter(image_size),
            image_count
        })
    }
}

#[allow(dead_code)]
struct TrainingLabelSet<R: Read> {
    labels: SimpleDataIter<Label, R>,
    label_count: u32
}

impl<R: Read> TrainingLabelSet<R> {
    fn try_from(mut input: R) -> Result<Self> {
        verify_magic(&mut input, DataKind::Label)?;
        let label_count: u32 = input.read_be()?;
        Ok(TrainingLabelSet {
            labels: input.data_iter(()),
            label_count
        })
    }
}

pub struct TrainingDataset<I: Read, L: Read> {
    images: TrainingImageSet<I>,
    labels: TrainingLabelSet<L>,
    read: u32
}

impl<I: Read, L: Read> TrainingDataset<I, L> {
    fn new(images: TrainingImageSet<I>, labels: TrainingLabelSet<L>) -> Self {
        Self {
            images,
            labels,
            read: 0
        }
    }

    pub fn from_readers(images: I, labels: L) -> Result<Self> {
        let images = TrainingImageSet::try_from(images)?;
        let labels = TrainingLabelSet::try_from(labels)?;

        if images.image_count != labels.label_count {
            return Err(ErrorKind::InvalidLabelCount {
                sample_count: images.image_count,
                label_count: labels.label_count
            })
        }

        Ok(Self::new(images, labels))
    }

    pub fn size(&self) -> u32 {
        self.images.image_count
    }
}

impl<I: Read, L: Read> Iterator for TrainingDataset<I, L> {
    type Item = Result<LabeledTrainingData>;

    fn next(&mut self) -> Option<Self::Item> {
        return if self.read < self.size() {
            let image = match self.images.images.next() {
                Some(Ok(image)) => image,
                Some(Err(err)) => return Some(Err(err.into())),
                None => return None
            };

            let label = match self.labels.labels.next() {
                Some(Ok(label)) => label,
                Some(Err(err)) => return Some(Err(err.into())),
                None => return None
            };

            self.read += 1;
            Some(Ok(LabeledTrainingData::new(image, label)))
        } else {
            None
        }
    }
}