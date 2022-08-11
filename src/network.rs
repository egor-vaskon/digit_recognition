use std::alloc::LayoutError;
use std::fs::{File, OpenOptions, write};
use std::{io, mem};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use nalgebra::{ArrayStorage, Const, DMatrix, DVector, Dynamic, max, OMatrix, OVector, U10, Unit, VecStorage, Vector, Vector3};
use rand_distr::{Binomial, Normal, StandardNormal};
use rand::{Rng, thread_rng};
use rand::rngs::ThreadRng;
use rand::distributions::{Bernoulli, Distribution};
use serde::{Serialize, Deserialize};
use std::f64::consts::E;
use std::iter::zip;
use std::ops::{AddAssign, Mul, MulAssign, SubAssign};
use druid::kurbo::Shape;
use druid::piet::util::resolve_range;
use thiserror::Error;
use typed_io::Endianness::LE;
use crate::launch;

#[derive(Error, Debug)]
pub enum ErrorKind {
    #[error("cannot read neural network save file ({0})")]
    CannotLoadNeuralNetwork(#[source] io::Error),

    #[error("cannot save neural network ({0})")]
    CannotSaveNeuralNetwork(#[source] io::Error),

    #[error("cannot parse neural network save file ({0})")]
    CannotParseNeuralNetworkFile(#[from] serde_json::Error)
}

pub type Result<T> = std::result::Result<T, ErrorKind>;

pub const INPUT_LAYER_SIZE: usize = 28*28;
pub const OUTPUT_LAYER_SIZE: usize = 10;
const HIDDEN_LAYER_SIZE: usize = 20;

const PRECISION: f64 = 1e-8;

const MIN_WEIGHT_OR_BIAS: f64 = -1.0 + PRECISION;
const MAX_WEIGHT_OR_BIAS: f64 = 1.0 - PRECISION;

const LEARNING_RATE: f64 = 0.1;
const ACCURACY: f64 = 0.01;

#[inline(always)]
fn relu(x: f64) -> f64 {
    /*return if x >= 0.0 {
        x
    } else {
        0.1*x
    }*/

    1.0 / (1.0 + E.powf(-x))
}

#[inline(always)]
fn relu_prime(x: f64) -> f64 {
    /*return if x > 0.0 {
        1.0
    } else {
        0.1
    }*/
    let val = relu(x);
    val*(1.0-val)
}

fn cross_entropy_loss(out: &DVector<f64>, expected: &DVector<f64>) -> f64 {
    eprintln!("out: {:.10}", out);
    eprintln!("expected: {:.10}", expected);

    let mut result = 0.0;
    for (i, out_i) in zip(expected.iter(), out.iter()) {
        let (i, out_i) = (*i, *out_i);
        result += i*out_i.log2();
    }

    -result
}

fn softmax(vec: &mut DVector<f64>) {
    let mx = vec.max();
    vec.apply(|x| *x = E.powf(*x-mx));

    let exp_sum = vec.sum();
    vec.apply(|x| {
        *x /= exp_sum
    });
}

fn softmax_prime(vec: &DVector<f64>) -> DVector<f64> {
    let mut result = vec.clone_owned();
    softmax(&mut result);

    result.apply(|x| *x = *x*(1.0-*x));
    result
}

#[derive(Serialize, Deserialize)]
struct Layer {
    weights: DMatrix<f64>,
    biases: DVector<f64>
}

impl Layer {
    fn new_untrained(rng: &mut impl Rng,
                     weight_distr: &impl Distribution<f64>,
                     bias_distr: &impl Distribution<f64>,
                     prev_dim: usize,
                     dim: usize) -> Layer {
        let b_distr = Bernoulli::new(0.5).unwrap();

        let weights =
            DMatrix::from_fn(dim, prev_dim, |_, _| rng.sample(weight_distr));

        let biases =
            DVector::from_fn(dim, |_, _| {
                let x = rng.sample(bias_distr);
                return if rng.sample(&b_distr) {
                    -x
                } else {
                    x
                }
            });

        Layer {
            weights,
            biases
        }
    }

    fn dim(&self) -> usize {
        self.biases.nrows()
    }
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>
}

struct NetworkResult {
    result: DVector<f64>,
    activations: Vec<DVector<f64>>,
    derivatives: Vec<DVector<f64>>
}

impl NeuralNetwork {
    pub fn load<P: AsRef<Path>>(file: P) -> Result<NeuralNetwork> {
        let file = File::open(file)
            .map_err(|err| ErrorKind::CannotLoadNeuralNetwork(err))?;

        let reader = BufReader::new(file);
        let network: NeuralNetwork = serde_json::from_reader(reader)?;

        Ok(network)
    }

    pub fn save<P: AsRef<Path>>(&self, file: P) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(file)
            .map_err(|err| ErrorKind::CannotSaveNeuralNetwork(err))?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;

        Ok(())
    }

    pub fn new_untrained() -> NeuralNetwork {
        let mut rng = thread_rng();
        let weight_distr = Normal::new(0.0, 0.1).unwrap();
        let bias_distr = Normal::new(0.0, 0.1).unwrap();

        let layers = vec![
            Layer::new_untrained(&mut rng, &weight_distr, &bias_distr, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            //Layer::new_untrained(&mut rng, &weight_distr, &bias_distr, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            Layer::new_untrained(&mut rng, &weight_distr, &bias_distr, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        ];

        NeuralNetwork {
            layers
        }
    }

    pub fn compute(&self, input: DVector<f64>) -> DVector<f64> {
        self.compute_ex(input).result
    }

    fn compute_ex(&self, input: DVector<f64>) -> NetworkResult {
        if input.len() != INPUT_LAYER_SIZE {
            panic!("this network requires input to be a {}-dimensional column vector", INPUT_LAYER_SIZE)
        }

        let mut result = NetworkResult {
            result: input,
            activations: Vec::new(),
            derivatives: Vec::new()
        };

        result.activations.push(result.result.clone_owned());

        for (i, layer) in self.layers.iter().enumerate() {
            let mut tmp = DVector::zeros(layer.dim());

            layer.weights.mul_to(&result.result, &mut tmp);
            tmp += &layer.biases;

            if i != 1 {
                result.derivatives.push(tmp.map(|x| relu_prime(x)));
                tmp.apply(|x| *x = relu(*x));
                result.activations.push(tmp.clone_owned());
            }

            //println!("tmp{}: {:.5}", i, tmp);
            result.result = tmp;
        }

        result.derivatives.push(softmax_prime(&result.result));
        softmax(&mut result.result);

        println!("result: {:.2}", &result.result);

        result
    }

    pub fn train(&mut self,
                 input: DVector<f64>,
                 target: &DVector<f64>) {
        let result = self.compute_ex(input.clone_owned());

        let error = cross_entropy_loss(&result.result, target);
        println!("error: {}", error);

        let mut local_gradients = vec![
            result.result
        ];

        local_gradients[0] -= target;
        local_gradients[0].component_mul_assign(&result.derivatives[1]);

        for i in (0..1).rev() {
            let layer_size = self.layers[i+1].weights.ncols();
            let prev_gradient = local_gradients.last().unwrap();

            let mut local_gradient = DVector::zeros(layer_size);

            // weights
            self.layers[i+1].weights.tr_mul_to(prev_gradient, &mut local_gradient);

            // biases
            for (j, bias) in self.layers[i+1].biases.column(0).iter().enumerate() {
                local_gradient[j] += bias;
            }

            local_gradient.component_mul_assign(&result.derivatives[i]);

            local_gradients.push(local_gradient);
        }

        local_gradients.reverse();

        /*for (i, layer) in self.layers.iter_mut().enumerate() {
            // update weights
            let prev_activation = &result.activations[i];
            for (k, mut column) in layer.weights.column_iter_mut().enumerate()  {
                for (j, mut element) in column.iter_mut().enumerate() {
                    *element -=
                        LEARNING_RATE * prev_activation[k] * local_gradients[i][j];
                }
            }

            // update biases
            layer.biases.sub_assign(local_gradients[i].clone_owned() * LEARNING_RATE);
        }*/
    }
}