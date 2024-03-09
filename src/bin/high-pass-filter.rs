use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
struct Entry {
    #[serde(rename = "freq")]
    frequency: f64,

    u_in: f64,
    #[serde(rename = "u_in_res")]
    u_in_resolution: f64,

    u_out: f64,
    #[serde(rename = "u_out_res")]
    u_out_resolution: f64,

    u_phase_shift: f64,
    #[serde(rename = "u_phase_shift_res")]
    u_phase_shift_resolution: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let csv: Vec<Entry> = csv::Reader::from_path("data/high-pass-filter.csv")?
        .deserialize()
        .collect::<Result<_, _>>()?;

    println!("{:?}", csv);
    Ok(())
}
