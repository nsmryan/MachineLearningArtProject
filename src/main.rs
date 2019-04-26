extern crate fann;
extern crate libc;
extern crate imageproc;
extern crate rand;

use std::boxed::Box;
//use libc::c_uint;

use image::ImageBuffer;
use image::Rgb;

use imageproc::drawing::*;

use rand::prelude::*;

use fann::{ActivationFunc, Fann, TrainAlgorithm};
use fann::TrainData;


const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const NUM_EXAMPLES: u32 = 10;

type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;

fn main() {
    let mut images: Vec<Vec<u8>> = Vec::new();

    let mut rng = rand::thread_rng();

    for index in 0..NUM_EXAMPLES {
        let mut image = Image::new(WIDTH, HEIGHT);

        let x: i32 = (rng.gen::<f32>() * WIDTH as f32) as i32;
        let y: i32 = (rng.gen::<f32>() * HEIGHT as f32) as i32;
        draw_filled_ellipse_mut(&mut image, (x, y), 25, 25, Rgb { data: [128, 0, 0] });

        image.save(format!("./image_{}.png", index)).unwrap();

        images.push(image.into_vec());
    }

    let mut fann = Fann::new(&[WIDTH*HEIGHT, 2, WIDTH*HEIGHT]).unwrap();

    fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
    fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);

    fann.set_train_algorithm(TrainAlgorithm::Quickprop(Default::default()));


    let max_epochs = 50000;
    let epochs_between_reports = 1000;
    let desired_error = 0.001;

    let training_data_callback = move |index| {
        let vec: &Vec<u8> = &images[index as usize];
        let bytes = image_to_vec(vec);
        (bytes.clone(), bytes.clone())
    };

    let training_data = TrainData::from_callback(NUM_EXAMPLES,
                                                 WIDTH*HEIGHT,
                                                 WIDTH*HEIGHT,
                                                 Box::new(training_data_callback)).unwrap();

    fann.on_data(&training_data)
        .with_reports(epochs_between_reports)
        .train(max_epochs, desired_error).unwrap();
}

fn image_to_vec(image: &Vec<u8>) -> Vec<f32> {
    image.iter().enumerate() 
                .filter(|(index, val)| index % 3 == 0)
                .map(|(index, val)| (*val as f32) / 256.0)
                .collect::<Vec<f32>>()
}

