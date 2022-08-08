use std::fmt::{Debug, Display, Formatter};
use std::io;
use std::io::Read;
use typed_io::TypedRead;
use thiserror::Error;
use crate::data::Image;
use crate::io_ext::{IntoDataIter, ReadFromBytes, SimpleDataIter};

#[cfg(test)]
mod test {
    use std::io::Cursor;
    use super::*;

    const IMAGE_HEADER_SIZE: usize =
          4 /* width (4 be bytes) */
        + 4 /* height (4 be bytes) */;

    const IMAGE_WIDTH: u32 = 2;
    const IMAGE_HEIGHT: u32 = 2;

    const IMAGE_PIXELS_SIZE: usize =
        1 /* pixel size */  * IMAGE_WIDTH as usize /* width */ * IMAGE_HEIGHT as usize /* height */;

    const IMAGE_SIZE: usize = IMAGE_HEADER_SIZE + IMAGE_PIXELS_SIZE;
    const LABEL_SIZE: usize = 1 /* digit (1 byte) */;

    const HEADER_SIZE: usize =
              MAGIC_SIZE /* magic (4 bytes) */
            + 4 /* number of items (e.g. images or labels - 4 be bytes) */;

    const NUM_ITEMS: u32 = 2;

    const IMAGES_SIZE: usize = HEADER_SIZE + IMAGE_SIZE * NUM_ITEMS as usize;
    const LABELS_SIZE: usize = HEADER_SIZE + LABEL_SIZE * NUM_ITEMS as usize;

    const LABEL_DIGIT: u8 = 7;

    static IMAGES_BYTES: [u8; IMAGES_SIZE] = [
        // magic
        MAGIC_BYTES[0], MAGIC_BYTES[1], MAGIC_BYTES[2], MAGIC_BYTES[3],

        // number of images (in be)
        0x00, 0x00, 0x00, 0x02,

        // begin of the 1st image

        0x00, 0x00, 0x00, 0x02, // width (in be)
        0x00, 0x00, 0x00, 0x02, // height (in be)

        // pixels (row major)
        0x01, 0x02,
        0x03, 0x04,

        // end of the 1st image

        // begin of the 2nd image

        0x00, 0x00, 0x00, 0x02, // width (in be)
        0x00, 0x00, 0x00, 0x02, // height (in be)

        // pixels (row major)
        0x01, 0x02,
        0x03, 0x04,

        // end of the 2nd image
    ];

    const LABELS_BYTES: [u8; LABELS_SIZE] = [
        // magic
        MAGIC_BYTES[0], MAGIC_BYTES[1], MAGIC_BYTES[2], MAGIC_BYTES[3],

        // number of labels (in be)
        0x00, 0x00, 0x00, 0x02,

        // 1st label
        0x07,

        // 2nd label
        0x07
    ];

    fn image_bytes() -> &'static [u8] {
        &IMAGES_BYTES[HEADER_SIZE..HEADER_SIZE+IMAGE_SIZE]
    }

    fn label_bytes() -> &'static [u8] {
        &LABELS_BYTES[HEADER_SIZE..HEADER_SIZE+LABEL_SIZE]
    }

    fn image_pixels() -> &'static [u8] {
        const START: usize = HEADER_SIZE+IMAGE_HEADER_SIZE;
        &IMAGES_BYTES[START..START+IMAGE_PIXELS_SIZE]
    }

    #[test]
    fn test_read_image() {
        let image =
            Image::read_from_bytes(&mut Cursor::new(image_bytes()))
                .unwrap();

        assert_image_valid(image);
    }

    #[test]
    fn test_read_label() {
        let label =
            Label::read_from_bytes(&mut Cursor::new(label_bytes()))
                .unwrap();

        assert_label_valid(label);
    }

    #[test]
    fn test_read_training_sample() {
        test_read_image();
        test_read_label();
    }

    #[test]
    fn test_read_image_set() {
        let mut images = Cursor::new(IMAGES_BYTES);
        let mut image_set =
            TrainingImageSet::try_from(images).unwrap();

        assert_eq!(image_set.image_count, NUM_ITEMS);

        for _ in 0..image_set.image_count {
            let image = image_set.images.next().unwrap().unwrap();
            assert_image_valid(image);
        }

        assert!(image_set.images.next().is_none())
    }

    #[test]
    fn test_read_label_set() {
        let mut labels = Cursor::new(LABELS_BYTES);
        let mut label_set =
            TrainingLabelSet::try_from(labels).unwrap();

        assert_eq!(label_set.label_count, NUM_ITEMS);

        for _ in 0..label_set.label_count {
            let label = label_set.labels.next().unwrap().unwrap();
            assert_label_valid(label);
        }

        assert!(label_set.labels.next().is_none())
    }

    #[test]
    fn test_read_training_dataset() {
        let mut images = Cursor::new(IMAGES_BYTES);
        let mut labels = Cursor::new(LABELS_BYTES);

        let mut dataset =
            TrainingDataset::from_readers(images, labels).unwrap();

        assert_eq!(dataset.read, 0);

        for i in 1..=NUM_ITEMS {
            assert_valid(dataset.next().unwrap().unwrap());
            assert_eq!(dataset.read, i);
        }

        assert!(dataset.next().is_none());
    }

    fn assert_valid(sample: LabeledTrainingData) {
        assert_label_valid(sample.label);
        assert_image_valid(sample.image);
    }

    fn assert_label_valid(label: Label) {
        assert_eq!(label, Label::new(LABEL_DIGIT));
    }

    fn assert_image_valid(image: Image) {
        assert_eq!(image,
                   Image::builder()
                       .with_size(IMAGE_WIDTH, IMAGE_HEIGHT)
                       .with_pixels_row_major(image_pixels())
                       .build());
    }
}

const MAGIC_SIZE: usize = 4;
const MAGIC_BYTES: [u8; MAGIC_SIZE] = [0x00, 0x00, 0x08, 0x01];
const MAGIC: u32 = u32::from_be_bytes(MAGIC_BYTES);

#[derive(Debug)]
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
    #[error("expected magic number ({}), found {magic:#x} (in {dataset_kind})", MAGIC)]
    MagicNotFound {
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

impl ReadFromBytes for Image {
    type Error = io::Error;

    fn read_from_bytes(input: &mut impl Read) -> std::result::Result<Self, Self::Error> {
        let width: u32 = input.read_be()?;
        let height: u32 = input.read_be()?;
        let area = (width as usize) * (height as usize);

        let mut pixels = vec![0; area];
        input.read_exact(&mut pixels)?;

        Ok(Image::builder()
            .with_size(width, height)
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
}

impl ReadFromBytes for Label {
    type Error = io::Error;

    fn read_from_bytes(input: &mut impl Read) -> std::result::Result<Self, Self::Error> {
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
}

fn verify_magic<R: Read>(input: &mut R, data_kind: DataKind) -> Result<()> {
    let magic = input.read_be::<u32>()?;
    return if magic == MAGIC {
        Ok(())
    } else {
        Err(ErrorKind::MagicNotFound {
            magic,
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
        Ok(TrainingImageSet {
            images: input.data_iter(),
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
            labels: input.data_iter(),
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

    fn len(&self) -> u32 {
        self.images.image_count
    }
}

impl<I: Read, L: Read> Iterator for TrainingDataset<I, L> {
    type Item = Result<LabeledTrainingData>;

    fn next(&mut self) -> Option<Self::Item> {
        return if self.read < self.len() {
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