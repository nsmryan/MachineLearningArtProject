extern crate fann;
extern crate libc;
extern crate imageproc;

use std::boxed::Box;
use libc::c_uint;

use image::ImageBuffer;
use image::Rgb;

use imageproc::drawing::*;

use fann::{ActivationFunc, Fann, TrainAlgorithm, QuickpropParams};
use fann::TrainData;


const WIDTH: i32 = 512;
const HEIGHT: i32 = 512;

fn main() {
    let mut fann = Fann::new(&[2, 3, 1]).unwrap();

    fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
    fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);

    fann.set_train_algorithm(TrainAlgorithm::Quickprop(Default::default()));

    let max_epochs = 50000;
    let epochs_between_reports = 1000;
    let desired_error = 0.001;

    let training_data = TrainData::from_callback(4, 2, 1, Box::new(training_data_callback)).unwrap();

    fann.on_data(&training_data)
        .with_reports(epochs_between_reports)
        .train(max_epochs, desired_error).unwrap();

    println!("{:?}", fann.run(&[1.0, 1.0]).unwrap());
    println!("{:?}", fann.run(&[0.0, 1.0]).unwrap());
    println!("{:?}", fann.run(&[1.0, 0.0]).unwrap());
    println!("{:?}", fann.run(&[1.0, 0.0]).unwrap());

    let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(WIDTH, HEIGHT);

    draw_filled_ellipse_mut(&mut image, (100, 100), 25, 25, Rgb { data: [128, 50, 0] });

    image.save("./image.png");
}

fn training_data_callback(index: c_uint) -> (Vec<f32>, Vec<f32>) {
    let examples = vec!((vec!(1.0, 1.0), vec!(1.0)),
                        (vec!(1.0, 0.0), vec!(1.0)),
                        (vec!(0.0, 1.0), vec!(1.0)),
                        (vec!(0.0, 0.0), vec!(0.0)));

    examples[index as usize].clone()
}
