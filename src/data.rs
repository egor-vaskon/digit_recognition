
#[derive(Debug, PartialEq)]
pub struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>
}

impl Image {
    pub fn builder() -> ImageBuilder {
        ImageBuilder {
            width: 0,
            height: 0,
            pixels: vec![]
        }
    }
}

pub struct ImageBuilder {
    width: u32,
    height: u32,
    pixels: Vec<u8>
}

impl ImageBuilder {
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        (self.width, self.height) = (width, height);
        self
    }

    pub fn with_pixels_row_major(mut self, pixels: impl Into<Vec<u8>>) -> Self {
        self.pixels = pixels.into();
        self
    }

    pub fn build(self) -> Image {
        if self.pixels.len() != (self.width as usize) * (self.height as usize) {
            panic!("incorrect image data size")
        }

        Image {
            width: self.width,
            height: self.height,
            pixels: self.pixels
        }
    }
}