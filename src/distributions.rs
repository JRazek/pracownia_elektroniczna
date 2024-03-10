use super::integral::Integral;

pub struct DistributionStats {
    expected_value: f64,
    variance: f64,
}

pub struct CompactlySupportedDistribution<F, PDF> {
    pub support: (f64, f64),
    pub f: F,
    pub pdf: PDF,
}

impl<F, PDF> CompactlySupportedDistribution<F, PDF>
where
    F: Fn(f64) -> f64,
    PDF: Fn(f64) -> f64,
{
    pub fn new(support: (f64, f64), f: F, pdf: PDF) -> Self {
        Self { support, f, pdf }
    }

    pub fn transform_x<G>(
        &self,
        g: G,
    ) -> CompactlySupportedDistribution<impl Fn(f64) -> f64 + Clone, PDF>
    where
        G: Fn(f64) -> f64 + Clone,
        PDF: Clone,
        F: Clone,
    {
        let support = self.support;
        let pdf = self.pdf.clone();
        let f = self.f.clone();

        CompactlySupportedDistribution::new(support, move |x| g(f(x)), pdf)
    }

    pub fn expected_value<I>(&self, integral: &I) -> f64
    where
        I: Integral,
        PDF: Clone,
        F: Clone,
    {
        let pdf = self.pdf.clone();
        let f = self.f.clone();

        let integrand = |x: f64| f(x) * pdf(x);
        integral.integrate(integrand, self.support)
    }

    pub fn calculate_distribution<I>(&self, integral: &I) -> DistributionStats
    where
        I: Integral,
        PDF: Clone,
        F: Clone,
    {
        let expected_value = self.expected_value(integral);

        let transformed_distribution = self.transform_x(|x: f64| x.powi(2) as f64);
        let transformed_expected_value = transformed_distribution.expected_value(integral);

        let variance = transformed_expected_value - expected_value.powi(2);

        DistributionStats {
            expected_value,
            variance,
        }
    }
}

#[derive(Debug)]
struct EmptyRangeError;

fn uniform_distribution_pdf(
    support: (f64, f64),
) -> Result<
    CompactlySupportedDistribution<
        impl Fn(f64) -> f64 + Copy + Clone + 'static,
        impl Fn(f64) -> f64 + Copy + Clone + 'static,
    >,
    EmptyRangeError,
> {
    let (a, b) = support.clone();
    let range = a..b;

    if range.is_empty() {
        Err(EmptyRangeError)?;
    };

    let pdf = move |x: f64| {
        let range = a..b;
        (if range.contains(&x) { 1. } else { 0. }) / (b - a) as f64
    };

    let identity = |x: f64| x;

    Ok(CompactlySupportedDistribution {
        support,
        f: identity,
        pdf,
    })
}

fn uniform_distribution_stats(support: (f64, f64)) -> Result<DistributionStats, EmptyRangeError> {
    let (a, b) = support;
    let range = a..b;

    if range.is_empty() {
        Err(EmptyRangeError)?;
    };

    let mean = (a + b) / 2.;
    let variance = (b - a).powi(2) / 12.;

    Ok(DistributionStats {
        expected_value: mean,
        variance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::integral::RiemannSum;

    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_distribution() {
        let uniform = uniform_distribution_pdf((0., 1.)).unwrap();
        let transformed_uniform = uniform.transform_x(|x: f64| x as f64);

        let stats = transformed_uniform.calculate_distribution(&RiemannSum { n: 1000 });

        let defined_stats = uniform_distribution_stats((0., 1.)).unwrap();

        assert_relative_eq!(
            stats.expected_value,
            defined_stats.expected_value,
            epsilon = 1e-2
        );
        assert_relative_eq!(stats.variance, defined_stats.variance, epsilon = 1e-2);
    }
}
