

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32
}

impl ImageSize {
    pub fn square(dimension: u32) -> ImageSize {
        ImageSize {
            width: dimension,
            height: dimension
        }
    }

    pub fn area(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

#[derive(Debug, PartialEq)]
pub struct Image {
    size: ImageSize,
    pixels: Vec<u8>
}

impl Image {
    pub fn builder() -> ImageBuilder {
        ImageBuilder {
            size: ImageSize::default(),
            pixels: vec![]
        }
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

pub struct ImageBuilder {
    size: ImageSize,
    pixels: Vec<u8>
}

impl ImageBuilder {
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = size;
        self
    }

    pub fn with_pixels_row_major(mut self, pixels: impl Into<Vec<u8>>) -> Self {
        self.pixels = pixels.into();
        self
    }

    pub fn build(self) -> Image {
        if self.pixels.len() != self.size.area() {
            panic!("incorrect image data size")
        }

        Image {
            size: self.size,
            pixels: self.pixels
        }
    }
}