use digit_recognition::{ErrorKind, launch};

fn main() {
    if let Some(err) = launch().err() {
        if let ErrorKind::CliError(err) = err {
            eprintln!("{}", err)
        } else {
            eprintln!("Error: {}", err);
        }
    }
}