use approx::{relative_eq, relative_ne};
use core::f64;
use plotters::prelude::*;
use serde::Deserialize;
use std::error::Error;

const W_START: f64 = 2. * std::f64::consts::PI * 100.;
const H: f64 = 1e-6;

fn alpha_from_w_rc(w: f64, rc: f64) -> f64 {
    let rwc = w * rc;

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

fn alpha_line_series<S>(rc: f64, style: S) -> LineSeries<SVGBackend<'static>, (f64, f64)>
where
    S: Into<ShapeStyle>,
{
    let alpha: LineSeries<SVGBackend<'_>, (f64, f64)> = LineSeries::new(
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
            .map(|w| (w, alpha_from_w_rc(w, rc))),
        style,
    );

    alpha
}

fn alpha_from_voltages(u_in: f64, u_out: f64) -> f64 {
    (u_out / u_in).abs()
}

fn beta(u_in: f64, u_out: f64) -> f64 {
    let alpha_i = alpha_from_voltages(u_in, u_out);
    let beta_i = alpha_i / (1. - alpha_i.powi(2)).sqrt();

    beta_i
}

fn fit_rc(entries: &[Measurement]) -> f64 {
    let (w, beta): (Vec<_>, Vec<_>) = entries
        .into_iter()
        .filter(|e| {
            let u_in = e.u_in * e.u_in_resolution;
            let u_out = e.u_out * e.u_out_resolution;
            let alpha_i = alpha_from_voltages(u_in, u_out);

            relative_ne!(alpha_i, 1.)
        })
        .map(|e| {
            let w_i = e.w;

            let u_in = e.u_in * e.u_in_resolution;
            let u_out = e.u_out * e.u_out_resolution;

            let beta_i = beta(u_in, u_out);

            //assuming w_i is with no uncertainty
            (w_i, beta_i)
        })
        .unzip();

    let model_rc: f64 = w
        .iter()
        .zip(beta.iter())
        .map(|(w_i, b_i)| w_i * b_i)
        .sum::<f64>()
        / w.iter().map(|w_i| w_i.powi(2)).sum::<f64>();

    model_rc
}

fn main() -> Result<(), Box<dyn Error>> {
    const R: f64 = 1e3;
    const C: f64 = 100e-9;

    let drawing_area =
        SVGBackend::new("plots/high-pass-filter-alpha-ref.svg", (800, 600)).into_drawing_area();

    let mut chart_builder = ChartBuilder::on(&drawing_area);

    let mut chart_context = chart_builder
        .caption("α(ω)", ("Arial", 20))
        .margin(40)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d((W_START..1e7).log_scale(), 0f64..1.2)?;

    chart_context
        .configure_mesh()
        .x_labels(20)
        .y_labels(20)
        .x_desc("ω [rad/s]")
        .y_desc("α")
        .x_label_formatter(&|x| format!("{:.0e}", x))
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let entries: Vec<Measurement> = csv::Reader::from_path("data/high-pass-filter.csv")?
        .deserialize()
        .collect::<Result<_, _>>()?;

    let fitted_rc = fit_rc(&entries);
    println!("fitted_rc: {fitted_rc}");

    let theoretical_alpha = alpha_line_series(R * C, &GREEN);
    let fit_alpha_plot = alpha_line_series(fitted_rc, &RED);

    chart_context
        .draw_series(fit_alpha_plot)?
        .label("Dopasowany model");

    chart_context
        .draw_series(entries.iter().map(|e| {
            let u_in = e.u_in * e.u_in_resolution;
            let u_out = e.u_out * e.u_out_resolution;

            let u_in_uncertainty = 0.2 * e.u_in_resolution / f64::sqrt(3.);
            let u_out_uncertainty = 0.2 * e.u_out_resolution / f64::sqrt(3.);

            let alpha_i = alpha_from_voltages(u_in, u_out);
            let alpha_i_uncertainty = ((u_out_uncertainty / u_in).powi(2)
                + (u_out / u_in.powi(2) * u_in_uncertainty).powi(2))
            .sqrt();

            ErrorBar::new_vertical(
                e.w,
                alpha_i - alpha_i_uncertainty / 2.,
                alpha_i,
                alpha_i + alpha_i_uncertainty / 2.,
                BLUE.filled(),
                5,
            )
        }))?
        .label("Pomiary");

    drawing_area.present()?;

    Ok(())
}
