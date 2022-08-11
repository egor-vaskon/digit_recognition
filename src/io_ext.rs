use std::io;
use std::io::Read;
use std::marker::PhantomData;

pub trait ReadFromBytes {
    type Error: From<io::Error>;
    type Config;

    fn read_from_bytes(input: &mut impl Read, config: &Self::Config) -> Result<Self, Self::Error>
        where Self: Sized;
}

pub trait ReadData {
    fn read_data<T: ReadFromBytes>(&mut self, config: &T::Config) -> Result<T, T::Error>
        where Self: Sized;
}

impl<R: Read> ReadData for R {
    fn read_data<T: ReadFromBytes>(&mut self, config: &T::Config) -> Result<T, T::Error>
        where Self: Sized
    {
        T::read_from_bytes(self, config)
    }
}

pub struct DataIter<T: ReadFromBytes, R: Read, F>
    where F: Fn(&Result<T, T::Error>) -> bool
{
    input: R,
    predicate: F,
    config: T::Config
}

impl<T: ReadFromBytes, R: Read, F> Iterator for DataIter<T, R, F>
    where F: Fn(&Result<T, T::Error>) -> bool
{
    type Item = Result<T, T::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.input.read_data::<T>(&self.config);
        return if !(self.predicate)(&item) {
            None
        } else {
            Some(item)
        }
    }
}

type PredicateFnPtr<T> = fn(&Result<T, io::Error>) -> bool;

fn stop_on_io_error<T>(item: &Result<T, io::Error>) -> bool {
    item.is_ok()
}

pub trait IntoDataIter<T: ReadFromBytes> : Read {
    type Predicate: Fn(&Result<T, T::Error>) -> bool;

    fn data_iter(self, config: T::Config) -> DataIter<T, Self, Self::Predicate>
        where Self: Sized;
}

impl<T, R> IntoDataIter<T> for R
    where
        T: ReadFromBytes<Error = io::Error>,
        R: Read
{
    type Predicate = PredicateFnPtr<T>;

    fn data_iter(self, config: T::Config) -> DataIter<T, Self, Self::Predicate>
        where Self: Sized
    {
        DataIter {
            input: self,
            predicate: stop_on_io_error,
            config
        }
    }
}

pub type SimpleDataIter<T, R> = DataIter<T, R, PredicateFnPtr<T>>;