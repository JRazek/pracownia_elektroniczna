use serde::Deserialize;
use std::error::Error;

use plotters::prelude::*;

fn alpha(w: f64, r: f64, c: f64) -> f64 {
    let rwc = r * w * c;

    rwc / (1. + rwc.powi(2)).sqrt()
}

#[derive(Deserialize, Debug)]
struct Measurement {
    w: f64,

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

struct WAlpha {
    w: f64,
    w_uncertainty: f64,
    alpha: f64,
    alpha_uncertainty: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    const R: f64 = 1e3;
    const C: f64 = 100e-9;

    const W_START: f64 = 2. * std::f64::consts::PI * 100.;

    let entries: Vec<Measurement> = csv::Reader::from_path("data/high-pass-filter.csv")?
        .deserialize()
        .collect::<Result<_, _>>()?;

    let drawing_area =
        SVGBackend::new("plots/high-pass-filter-alpha-ref.svg", (800, 600)).into_drawing_area();

    let mut chart_builder = ChartBuilder::on(&drawing_area);

    let mut chart_context = chart_builder
        .caption("α(ω)", ("Arial", 20))
        .margin(40)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d((W_START..1e7).log_scale(), 0f64..1.0)?;

    chart_context
        .configure_mesh()
        .x_labels(20)
        .y_labels(20)
        .x_desc("ω [rad/s]")
        .y_desc("α")
        .x_label_formatter(&|x| format!("{:.0e}", x))
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let theoretical_alpha: LineSeries<_, _> = LineSeries::new(
        [W_START]
            .into_iter()
            //w \in (1^3, 10^5)
            .chain(
                (10..=1000)
                    .map(|i| {
                        let w = i as f64 * 1e2;

                        w
                    })
                    //w \in (10^5, 10^5 + 10^7)
                    .chain((0..1000).map(|i| {
                        let w = i as f64 * 1e4 + 1e5;

                        w
                    })),
            )
            .map(|w| (w, alpha(w, R, C))),
        RED.filled(),
    );

//    let experimental_alpha_measurements: 

    chart_context
        .draw_series(theoretical_alpha)?
        .label("Oczekiwany model");

    drawing_area.present()?;

    Ok(())
}
