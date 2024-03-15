use dfdx::prelude::*;
use std::ops::Add;
use std::ops::Sub;

use std::borrow::Borrow;

pub trait Function<D, const N: usize>
where
    D: Device<f32>,
{
    fn eval<PTape, XTape>(
        &self,
        x_i: Tensor<Rank0, f32, D, XTape>,
        params: Tensor<Rank1<N>, f32, D, PTape>,
    ) -> Tensor<Rank0, f32, D, PTape>
    where
        PTape: Tape<f32, D> + Merge<XTape>,
        XTape: Tape<f32, D>;
}

fn sample_error<D, const N: usize, F, T>(
    dataset_xi_yi: (f32, f32),
    f: impl Borrow<F>,
    params: Tensor<Rank1<N>, f32, D, T>,
) -> Tensor<(), f32, D, T>
where
    D: Device<f32>,
    F: Function<D, N>,
    T: Tape<f32, D>,
{
    let (x, y) = dataset_xi_yi;

    let x = params.device().tensor(x);
    let y = params.device().tensor(y);

    let error_i = f.borrow().eval(x, params).sub(y).square();

    error_i
}

pub fn fit<D, const N: usize, F>(
    dataset_xi_yi: impl Iterator<Item = (f32, f32)> + Clone,
    f: F,
    training_iterations: usize,
    sgd_config: SgdConfig,
    mut initial: Tensor<Rank1<N>, f32, D>,
) -> Result<Tensor<Rank1<N>, f32, D>, Box<dyn std::error::Error>>
where
    D: Device<f32>,
    F: Function<D, N>,
{
    let mut sgd = Sgd::new(&initial, sgd_config);

    for _ in 0..training_iterations {
        let loss: Tensor<(), _, _, _> = dataset_xi_yi
            .clone()
            .map(|xy| sample_error::<_, N, F, OwnedTape<f32, D>>(xy, f.borrow(), initial.retaped()))
            .reduce(|t1, t2| t1.add(t2))
            .unwrap();

        let grads = loss.backward();

        sgd.update(&mut initial, &grads)?;
    }

    Ok(initial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    struct LinearFunction;

    impl<D> Function<D, 1> for LinearFunction
    where
        D: Device<f32>,
    {
        fn eval<PTape, XTape>(
            &self,
            x_i: Tensor<Rank0, f32, D, XTape>,
            params: Tensor<Rank1<1>, f32, D, PTape>,
        ) -> Tensor<Rank0, f32, D, PTape>
        where
            PTape: Tape<f32, D> + Merge<XTape>,
            XTape: Tape<f32, D>,
        {
            params.sum() * x_i
        }
    }

    #[test]
    fn test_linear() {
        let dev: AutoDevice = Default::default();

        let dataset = vec![(0f32, 0f32), (1f32, 2f32), (3f32, 6f32), (4f32, 8f32)];

        let params = dev.sample_normal();
        let f = LinearFunction;

        let params_fitted = fit(dataset.into_iter(), f, 100, SgdConfig::default(), params)
            .unwrap()
            .array()[0];

        assert_relative_eq!(params_fitted, 2f32);
    }
}
