use std::fmt::Display;

use colored::Colorize;

pub type Result<T, E = Error> = core::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    // Recoverable,
    Unrecoverable,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Guaranteed to return an Err
pub fn raise<S, T>(note: S, e: T) -> Result<()>
where
    S: Display,
    T: std::error::Error + Display,
{
    let s = format!("{note}: {}", e.to_string().red());
    log::error!("{s}");
    Err(Error::Unrecoverable)
}
