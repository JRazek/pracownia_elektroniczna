use super::integral::Integral;

pub struct DistributionStats {
    mean: f64,
    variance: f64,
}

pub struct CompactlySupportedPDF<F> {
    pub support: (f64, f64),
    pub pdf: F,
}

impl<PDF> CompactlySupportedPDF<PDF>
where
    PDF: Fn(f64) -> f64,
{
    pub fn new(support: (f64, f64), pdf: PDF) -> Self {
        Self { support, pdf }
    }

    pub fn transform<F>(&self, f: F) -> CompactlySupportedPDF<impl Fn(f64) -> f64 + Clone>
    where
        F: Fn(f64) -> f64 + Clone,
        PDF: Clone,
    {
        let support = self.support;
        let pdf = self.pdf.clone();

        CompactlySupportedPDF::new(support, move |x| pdf(f(x)))
    }

    pub fn calculate_distribution<I>(&self, integral: I) -> DistributionStats
    where
        I: Integral,
        PDF: Clone,
    {
        let mean = integral.integrate(&self.pdf, self.support);
        let transformed_pdf = self.transform(|x: f64| x.powi(2) as f64);
        let variance = integral.integrate(&transformed_pdf.pdf, self.support) - mean.powi(2);

        DistributionStats { mean, variance }
    }
}

#[derive(Debug)]
struct EmptyRangeError;

fn uniform_distribution_pdf(
    support: (f64, f64),
) -> Result<CompactlySupportedPDF<impl Fn(f64) -> f64 + Copy + 'static>, EmptyRangeError> {
    let (a, b) = support.clone();
    let range = a..b;

    if range.is_empty() {
        Err(EmptyRangeError)?;
    };

    let pdf = move |x: f64| {
        let range = a..b;
        (if range.contains(&x) { 1. } else { 0. }) / 2. * (b - a) as f64
    };

    Ok(CompactlySupportedPDF { support, pdf })
}

fn uniform_distribution_stats(support: (f64, f64)) -> Result<DistributionStats, EmptyRangeError> {
    let (a, b) = support;
    let range = a..b;

    if range.is_empty() {
        Err(EmptyRangeError)?;
    };

    let mean = (a + b) / 2.;
    let variance = (b - a).powi(2) / 12.;

    Ok(DistributionStats { mean, variance })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::integral::RiemannSum;

    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_distribution() {
        let uniform = uniform_distribution_pdf((0., 1.)).unwrap();
        let transformed_uniform = uniform.transform(|x: f64| x.powi(2) as f64);

        let stats = transformed_uniform.calculate_distribution(RiemannSum { n: 1000 });

        let defined_stats = uniform_distribution_stats((0., 1.)).unwrap();

        assert_relative_eq!(stats.mean, defined_stats.mean);
        assert_relative_eq!(stats.variance, defined_stats.variance);
    }
}
