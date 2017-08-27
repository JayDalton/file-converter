extern crate byteorder;
extern crate rustfft;
extern crate rand;
// extern crate num;

use std::io::{self, Cursor, Read, Write};
use std::error::Error;
use std::{env, fmt, process};
// use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::iter;
// use std::string;
// use std::option::Option;
    
use byteorder::{BigEndian, ReadBytesExt, LittleEndian, WriteBytesExt};

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
// use rustfft::algorithm::DFT;
// use rustfft::algorithm::FFT;

use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};
// use std::f32;
// use std::num;

fn main() {

    // call: FileToImage.raw 4320 4318 1
    let args: Vec<String> = env::args().skip(1).take(5).collect();

    if args.len() > 2 {

        let input_name = &args[0];
        let output_temp = format!("{}.tmp", &args[0]);
        let output_name = format!("{}.out", &args[0]);
        let file_magnitude = format!("{}.out", &args[0]);

        let img_rows: usize = args[1].trim().parse().expect("2nd arg no number");
        let img_cols: usize = args[2].trim().parse().expect("3rd arg no number");

        let swap: bool = if args.len() == 4 { true } else { false };

        if check_file_metadata(input_name, img_rows * img_cols) 
        {
            let content = get_file_content_u16(input_name, swap);

            // let mtx_u16 = get_short_matrix(&content, img_rows, img_cols);
            let mtx_f32 = get_float_matrix(&content, img_rows, img_cols);

            let mut complex = get_complex_matrix(&mtx_f32);
            let fourier = fft_complex_matrix(&mut complex);

            let magnitude = get_magnitude_from(&fourier);

            set_file_logging_f32(&file_magnitude, &magnitude);

            // set_file_content_u16(&output_temp, c);


        } else {
            println!("File not readable.");
        }

    } else {
        println!("Aufruf: path/to/image img_rows img_cols <swap>");
    }

}

fn get_magnitude_from(matrix: &Vec<Vec<Complex<f32>>>) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(matrix.len());
    for line in matrix {
        let mut output: Vec<f32> = Vec::with_capacity(line.len());
        for value in line {
            let x = (value.norm_sqr() as f32).sqrt();
            output.push(x);
        }
        result.push(output);
    }
    result
}

fn fft_complex_matrix(data: &mut Vec<Vec<Complex<f32>>>) -> Vec<Vec<Complex<f32>>> {
    let mut result: Vec<Vec<Complex<f32>>> = Vec::with_capacity(data.len());

    let mut planner = FFTplanner::new(false);

    for line in &mut *data {
        let fft = planner.plan_fft(line.len());
        let mut output: Vec<Complex<f32>> = vec![Complex::zero(); line.len()];
        fft.process(&mut *line, &mut output);
        result.push(output);
    }

    result
}

fn get_complex_matrix(data: &Vec<Vec<f32>>) -> Vec<Vec<Complex<f32>>> {
    let mut result: Vec<Vec<Complex<f32>>> = Vec::with_capacity(data.len());

    for row in data {
        let mut onerow: Vec<Complex<f32>> = Vec::with_capacity(row.len());
        for val in row { onerow.push(Complex {re: *val, im: 0f32}); }
        result.push(onerow);
    }

    result
}

fn get_short_matrix(data: &Vec<u16>, rows: usize, cols: usize) -> Vec<Vec<u16>> {
    assert!(rows * cols == data.len());
    let mut result: Vec<Vec<u16>> = Vec::with_capacity(data.len());

    for row in data.chunks(cols) {
        result.push(row.to_vec());
    }

    result
}


fn get_float_matrix(data: &Vec<u16>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    assert!(rows * cols == data.len());
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(data.len());

    for row in data.chunks(cols) {
        let mut onerow: Vec<f32> = Vec::with_capacity(row.len());
        for col in row { onerow.push(*col as f32); }
        result.push(onerow);
    }

    result
}

fn check_file_metadata(s: &str, l: usize) -> bool {
    let metadata = fs::metadata(s).expect("Can't read metadata");
    assert!(metadata.is_file());
    println!("2 * {} == {}", l, metadata.len());
    assert!(l * 2 == metadata.len() as usize);
    true
}

fn get_file_content_u8(s: &str) -> Vec<u8> {

    let mut contents: Vec<u8> = Vec::new();
    let mut in_file = File::open(s).expect("could not open file");
    let _ = in_file.read_to_end(&mut contents);
    contents
}

fn get_file_content_u16(s: &str, swap: bool) -> Vec<u16> {
    let content = get_file_content_u8(s);
    let mut reader = Cursor::new(&content);
    let mut converted: Vec<u16> = Vec::new();
    while reader.position() < content.len() as u64 {
        if swap {
            converted.push(reader.read_u16::<BigEndian>().unwrap());
        } else {
            converted.push(reader.read_u16::<LittleEndian>().unwrap());
        }
    }
    converted
}

fn set_file_logging_u16(name: &str, data: &Vec<Vec<u16>>) {
    let mut outfile = File::create(name).expect("could not create file");
    for line in data {
        for value in line {
            write!(outfile, "{};", value).expect("could not write to file");
            // outfile.write_fmt(format_args!("{}", value));
            // match write!(outfile, "{}", value) {
            //     Err(why) => { panic!("couldn't write to {}: {}", name, why.description()) }
            //     Ok(_) => println!("successfully wrote to {}", name),
            // }
        }
        writeln!(outfile, "").expect("could not write to file")
    }
}

fn set_file_logging_f32(name: &str, data: &Vec<Vec<f32>>) {
    let mut outfile = File::create(name).expect("could not create file");

    for line in data {
        for value in line {
            write!(outfile, "{:.4};", value).expect("could not write to file");
            // outfile.write_fmt(format_args!("{}", value));
            // match write!(outfile, "{}", value) {
            //     Err(why) => { panic!("couldn't write to {}: {}", name, why.description()) }
            //     Ok(_) => println!("successfully wrote to {}", name),
            // }
        }
        writeln!(outfile, "").expect("could not write to file")
    }
}

fn set_file_content_u16(name: &str, values: Vec<u16>) {
    let mut result: Vec<u8> = Vec::new();
    for n in values {
        let _ = result.write_u16::<LittleEndian>(n);
    }
    set_file_content_u8(name, result);
}    
 
fn set_file_content_u8(name: &str, values: Vec<u8>) {
    let mut outfile = File::create(name).expect("could not create file");
    match outfile.write_all(&values) {
        Err(why) => {panic!("couldn't write to {}: {}", name, why.description())},
        Ok(_) => println!("successfully wrote to {}", name),
    }
}    

    // // Create a path to the desired file
    // let inputPath = Path::new("hello.txt");
    // let inputDisplay = inputPath.display();

    // // Open the path in read-only mode, returns `io::Result<File>`
    // let mut file = match File::open(&inputPath) {
    //     // The `description` method of `io::Error` returns a string that
    //     // describes the error
    //     Err(why) => panic!("couldn't open {}: {}", inputDisplay, why.description()),
    //     Ok(file) => file,
    // };

    // // Read the file contents into a string, returns `io::Result<usize>`
    // let mut s = String::new();
    // let mut buf = [0u8, 1024];

    // let bytes_read = file.read(&mut buf).unwrap();



    // for byte in file.bytes() {

    // }

    // match file.read_to_string(&mut s) {
    //     Err(why) => panic!("couldn't read {}: {}", inputDisplay, why.description()),
    //     Ok(_) => print!("{} contains:\n{}", inputDisplay, s),
    // }

/////////////////////////////////

    // let outputPath = Path::new("out/lorem_ipsum.txt");
    // let outputDisplay = outputPath.display();

    // static LOREM_IPSUM: &'static str =
    //     "Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
    //     tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
    //     quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
    //     consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
    //     cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non
    //     proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

    // // Open a file in write-only mode, returns `io::Result<File>`
    // let mut file = match File::create(&outputPath) {
    //     Err(why) => panic!("couldn't create {}: {}", outputDisplay, why.description()),
    //     Ok(file) => file,
    // };

    // // Write the `LOREM_IPSUM` string to `file`, returns `io::Result<()>`
    // match file.write_all(LOREM_IPSUM.as_bytes()) {
    //     Err(why) => {panic!("couldn't write to {}: {}", outputDisplay, why.description())},
    //     Ok(_) => println!("successfully wrote to {}", outputDisplay),
    // }

// fn copy<P: AsRef<Path>>(infile: P, outfile: P) -> io::Result<()> {
//     let mut vec = Vec::new();
 
//     Ok(try!(File::open(infile)
//          .and_then(|mut i| i.read_to_end(&mut vec))
//          .and_then(|_| File::create(outfile))
//          .and_then(|mut o| o.write_all(&vec))))
// }
 
fn exit_err<T: fmt::Display>(msg: T, code: i32) -> ! {
    writeln!(&mut io::stderr(), "ERROR: {}", msg).expect("Could not write to stdout");
    process::exit(code);
}