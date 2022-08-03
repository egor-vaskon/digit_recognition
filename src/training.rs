use std::fmt::{Debug, Display, Formatter};
use std::io;
use std::io::Read;
use typed_io::TypedRead;
use thiserror::Error;

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
            Image::read_from(&mut Cursor::new(image_bytes()))
                .unwrap();

        assert_image_valid(image);
    }

    #[test]
    fn test_read_label() {
        let label =
            Label::read_from(&mut Cursor::new(label_bytes()))
                .unwrap();

        assert_label_valid(label);
    }

    #[test]
    fn test_read_training_sample() {
        // Sample is an image and label from 2 different files,
        // so we can avoid duplicating code from tests for respective types.
        test_read_image();
        test_read_label();
    }

    #[test]
    fn test_read_training_dataset() {
        let mut images = Cursor::new(IMAGES_BYTES);
        let mut labels = Cursor::new(LABELS_BYTES);

        let mut dataset =
            TrainingDataset::builder()
                .with_images(images)
                .with_labels(labels)
                .build().unwrap();

        assert_eq!(dataset.sample_count, NUM_ITEMS);

        assert_eq!(dataset.samples_read, 0);

        for i in 1..=NUM_ITEMS {
            assert_sample_valid(dataset.next().unwrap().unwrap());
            assert_eq!(dataset.samples_read, i);
        }

        assert!(dataset.next().is_none());
    }

    fn assert_sample_valid(sample: TrainingSample) {
        assert_label_valid(sample.label);
        assert_image_valid(sample.image);
    }

    fn assert_label_valid(label: Label) {
        assert_eq!(label, Label { digit: LABEL_DIGIT });
    }

    fn assert_image_valid(image: Image) {
        assert_eq!(image,
                   Image {
                       width: IMAGE_WIDTH,
                       height: IMAGE_HEIGHT,
                       pixels: image_pixels().to_vec()
                   });
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

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>
}

impl Image {
    fn read_from(input: &mut impl Read) -> Result<Image> {
        let width: u32 = input.read_be()?;
        let height: u32 = input.read_be()?;
        let area = (width as usize) * (height as usize);

        let mut pixels = vec![0u8; area];
        input.read_exact(&mut pixels)?;

        Ok(Image {
            width,
            height,
            pixels
        })
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Label {
    digit: u8
}

impl Label {
    fn read_from(input: &mut impl Read) -> Result<Label> {
        Ok(Label { digit: input.read_ne::<u8>()? })
    }
}

impl Label {
    pub fn digit(&self) -> u8 {
        self.digit
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct TrainingSample {
    image: Image,
    label: Label
}

impl TrainingSample {
    fn read_from(images: &mut impl Read,
                 labels: &mut impl Read) -> Result<TrainingSample> {
        let image = Image::read_from(images)?;
        let label = Label::read_from(labels)?;

        Ok(TrainingSample { image, label })
    }
}

pub struct TrainingDataset<I: Read, L: Read> {
    sample_count: u32,
    images: I,
    labels: L,
    samples_read: u32
}

impl<I: Read, L: Read> TrainingDataset<I, L> {
    pub fn builder() -> TrainingDatasetBuilder<I, L> {
        TrainingDatasetBuilder {
            images: None,
            labels: None
        }
    }
}

impl<I: Read, L: Read> Iterator for TrainingDataset<I, L> {
    type Item = Result<TrainingSample>;

    fn next(&mut self) -> Option<Self::Item> {
        return if self.samples_read == self.sample_count {
            None
        } else {
            let sample =
                TrainingSample::read_from(&mut self.images, &mut self.labels);
            self.samples_read += 1;

            Some(sample)
        }
    }
}

pub struct TrainingDatasetBuilder<I: Read, L: Read> {
    images: Option<I>,
    labels: Option<L>
}

impl<I: Read, L: Read> TrainingDatasetBuilder<I, L> {
    pub fn with_images(mut self, images: I) -> Self {
        self.images = Some(images);
        self
    }

    pub fn with_labels(mut self, labels: L) -> Self {
        self.labels = Some(labels);
        self
    }

    pub fn build(self) -> Result<TrainingDataset<I, L>> {
        let mut images = self.images.expect("images source not set");
        let mut labels = self.labels.expect("labels source not set");

        let magic = images.read_be::<u32>()?;
        if magic != MAGIC {
            return Err(ErrorKind::MagicNotFound {
                magic, dataset_kind: DataKind::Image
            });
        }

        let magic = labels.read_be::<u32>()?;
        if magic != MAGIC {
            return Err(ErrorKind::MagicNotFound {
                magic, dataset_kind: DataKind::Label
            });
        }

        let image_count: u32 = images.read_be()?;
        let label_count: u32 = labels.read_be()?;

        return if image_count == label_count {
            Ok(TrainingDataset {
                sample_count: image_count,
                images,
                labels,
                samples_read: 0
            })
        } else {
            Err(ErrorKind::InvalidLabelCount {
                sample_count: image_count,
                label_count
            })
        }
    }
}