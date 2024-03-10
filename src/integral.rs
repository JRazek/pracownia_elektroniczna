use peroxide::numerical::integral::newton_cotes_quadrature;

pub trait Integral {
    fn integrate<F>(&self, f: F, support: (f64, f64)) -> f64
    where
        F: Fn(f64) -> f64;
}

pub struct NewtonCotesQuadrature {
    pub n: usize,
}

impl Integral for NewtonCotesQuadrature {
    fn integrate<F>(&self, f: F, support: (f64, f64)) -> f64
    where
        F: Fn(f64) -> f64,
    {
        newton_cotes_quadrature(f, self.n, support)
    }
}

pub struct RiemannSum {
    pub n: usize,
}

impl Integral for RiemannSum {
    fn integrate<F>(&self, f: F, support: (f64, f64)) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let (a, b) = support;
        let h = (b - a) / self.n as f64;
        let mut sum = 0f64;
        for i in 0..self.n {
            sum += f(a + h * (i as f64));
        }
        sum * h
    }
}
