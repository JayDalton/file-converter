extern crate rand;
extern crate rustfft;
extern crate num_cpus;
extern crate byteorder;

// Messagepack
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate rmp_serde as rmps;


use serde::{Deserialize, Serialize};
use rmps::{Deserializer, Serializer};

#[derive(Serialize, Deserialize, Debug)]
struct Matrix {
    rows: u16,
    cols: u16,
    data: Vec<f32>
}

// Threading
use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;

// Benchmark Timing
use std::time::{Duration, Instant};

use std::io::{self, Cursor, Read, Write};
use std::error::Error;
use std::{env, fmt, process};
// use std::io::prelude::*;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::iter;
    
use byteorder::{BigEndian, ReadBytesExt, LittleEndian, WriteBytesExt};

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_complex::Complex32;
use rustfft::num_traits::Zero;
// use rustfft::algorithm::DFT;
// use rustfft::algorithm::FFT;

// Randomize
use rand::{StdRng, SeedableRng};
use rand::distributions::{Normal, IndependentSample};


fn main() {

    // call: FileToImage.raw 4320 4318 1
    let args: Vec<String> = env::args().skip(1).take(5).collect();

    println!("Logical CPUs: {}", num_cpus::get());
    println!("Physical CPUs: {}", num_cpus::get_physical());

    test_multi_thread_fft();

    fft_complex_matrix_async();

    if args.len() > 2 {

        let input_name = &args[0];
        let fname_temp = &format!("{}.tmp", &args[0]);
        let fname_test = &format!("{}.tst", &args[0]);
        let fname_name = &format!("{}.out", &args[0]);
        let fname_magnitude = format!("{}.mag", &args[0]);

        let img_rows: usize = args[1].trim().parse().expect("2nd arg no number");
        let img_cols: usize = args[2].trim().parse().expect("3rd arg no number");

        let swap: bool = if args.len() == 4 { true } else { false };

        if check_file_metadata(&input_name, img_rows * img_cols) 
        {
            let mut now = Instant::now();

            let content = read_file_from_u16_to_f32(&input_name, swap);
            save_matrix_f32_to_msgpack(&format!("{}.rst.content.u16", input_name), &content, img_rows, img_cols);

            let complex = get_complex_matrix_2d(&content, img_rows, img_cols);
            let complex = get_complex_matrix(&content, img_rows, img_cols);
            // let complex = get_complex_matrix_old(&content);

            // let complex_mag = get_magnitude_from(&complex);
            // let flatten = get_flatten_matrix(&complex_mag);
            // save_matrix_f32_to_msgpack(&fname_test, &flatten, img_rows, img_cols);

            // now = dump_duration(now, "passed complex matrix");

            // let fourier = fft_complex_matrix(&complex);

            // now = dump_duration(now, "passed fft complex");

            // let magnitude = get_magnitude_from(&fourier);
            // let flatten = get_flatten_matrix(&magnitude);
            // save_matrix_f32_to_msgpack(&fname_name, &flatten, img_rows, img_cols);

            println!("the end ...");

        } else {
            println!("File not readable.");
        }

    } else {
        println!("Aufruf: path/to/image img_rows img_cols <swap>");
    }
}

fn dump_duration(now: std::time::Instant, label: &str) -> Instant {
    let duration = now.elapsed();
    println!("{}: {} secs", label, duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9);
    Instant::now()
}

fn save_matrix_f32_to_msgpack(fname: &str, values: &Vec<f32>, rows: usize, cols: usize) {
    debug_assert_eq!(values.len(), rows * cols);
    let now = Instant::now();
    if values.len() == rows * cols {
        let content = Matrix {
            rows: rows as u16,
            cols: cols as u16,
            data: values.clone()
        };
        let mut f = File::create(&fname).unwrap();
        content.serialize(&mut Serializer::new(&mut f)).unwrap();
    }
    dump_duration(now, &format!("msgpack matrix to {}", fname));
}

fn fft_complex_matrix_async() {

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("the"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            // thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }

}

fn test_multi_thread_fft() {
    let inverse = false;
    let mut planner = FFTplanner::new(inverse);
    let fft = planner.plan_fft(100);

    // let (tx, rx) = mpsc::channel();

    let threads: Vec<thread::JoinHandle<_>> = (0..2).map(|_| {
        let fft_copy = fft.clone();
        thread::spawn(move || {
            let mut signal = vec![Complex32::new(0.0, 0.0); 100];
            let mut spectrum = vec![Complex32::new(0.0, 0.0); 100];
            fft_copy.process(&mut signal, &mut spectrum);
        })
    }).collect();

    for thread in threads {
        thread.join().unwrap();
    }


}

fn fft_complex_matrix(matrix: &Vec<Vec<Complex<f32>>>) -> Vec<Vec<Complex<f32>>> {
    let mut result: Vec<Vec<Complex<f32>>> = Vec::with_capacity(matrix.len());

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(matrix.len());

    let mut counter = 0;

    for line in &*matrix {
        let mut input = line.clone();
        let mut output: Vec<Complex<f32>> = vec![Complex32::zero(); line.len()];
        fft.process(&mut input, &mut output);
        result.push(output);

        // println!("{}", counter++);
    }

    result
}

fn get_flatten_matrix(matrix: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::with_capacity(matrix.len());

    for line in matrix {
        result.extend(line);
    }

    result    
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

fn get_complex_matrix_2d(values: &Vec<f32>, rows: usize, cols: usize) -> Vec<Vec<Complex<f32>>> {
    debug_assert_eq!(values.len(), rows * cols);

    let now = Instant::now();
    let mut result: Vec<Vec<Complex<f32>>> = Vec::with_capacity(rows);

    for row in values.chunks(cols) {
        let mut onerow: Vec<Complex<f32>> = Vec::with_capacity(cols);
        for val in row { onerow.push(Complex {re: *val, im: 0f32}); }
        result.push(onerow);
    }
    dump_duration(now, "get_complex_matrix_2d");
    result
}

fn get_complex_matrix(values: &Vec<f32>, rows: usize, cols: usize) -> Vec<Complex32> {
    debug_assert_eq!(values.len(), rows * cols);

    let now = Instant::now();
    let mut result: Vec<Complex32> = Vec::with_capacity(rows * cols);
    for val in values { result.push(Complex32 {re: *val, im: 0f32}); }
    dump_duration(now, "get_complex_matrix");

    debug_assert_eq!(result.len(), rows * cols);
    result
}

fn get_complex_matrix_old(data: &Vec<Vec<f32>>) -> Vec<Vec<Complex<f32>>> {
    let now = Instant::now();
    let mut result: Vec<Vec<Complex<f32>>> = Vec::with_capacity(data.len());

    for row in data {
        let mut onerow: Vec<Complex<f32>> = Vec::with_capacity(row.len());
        for val in row { onerow.push(Complex {re: *val, im: 0f32}); }
        result.push(onerow);
    }
    dump_duration(now, "get_complex_matrix_old");
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

fn get_float_matrix(values: &Vec<f32>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    assert!(rows * cols == values.len());
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(values.len());
    result
}

fn get_float_matrix_old(data: &Vec<u16>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
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
    let now = Instant::now();
    let mut contents: Vec<u8> = Vec::new();
    let mut in_file = File::open(s).expect("could not open file");
    let _ = in_file.read_to_end(&mut contents);
    dump_duration(now, "read input file");
    contents
}

fn read_file_from_u16_to_f32(s: &str, swap: bool) -> Vec<f32> {
    let now = Instant::now();
    let content = get_file_content_u8(s);
    let mut reader = Cursor::new(&content);
    let mut converted: Vec<f32> = Vec::new();
    while reader.position() < content.len() as u64 {
        if swap {
            converted.push(reader.read_u16::<BigEndian>().unwrap() as f32);
        } else {
            converted.push(reader.read_u16::<LittleEndian>().unwrap() as f32);
        }
    }
    dump_duration(now, "convert file to f32");
    converted
}

fn get_file_content_from_u16(s: &str, swap: bool) -> Vec<u16> {
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

// fn write_samples(file: &str, samples: &Vec<f32>) {
//     let mut file = File::create(file).unwrap();
//     file.pack(samples.len() as u32).unwrap();
//     file.pack_all(&samples[..]).unwrap();
// }

// fn read_samples(file: &str) -> Vec<f32> {
//     let mut file = File::open(file).unwrap();
//     let num_samples : u32 = file.unpack().unwrap();
//     let mut samples : Vec<f32> = repeat(0f32).take(num_samples as usize).collect();
//     file.unpack_exact(&mut samples[..]).unwrap();
//     return samples;
// }


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

// write mtx to binary f32
fn set_file_logging_f32(name: &str, data: &Vec<Vec<f32>>) {

    let mut result: Vec<u8> = Vec::new();
    for line in data {
        for value in line {
            // let _ = result.write_f32::<LittleEndian>(value);
            
        }

    }
    set_file_content_u8(name, result);

    // let mut outfile = File::create(name).expect("could not create file");

    // for line in data {
    //     for value in line {
    //         write!(outfile, "{:.4};", value).expect("could not write to file");
    //         // outfile.write_fmt(format_args!("{}", value));
    //         // match write!(outfile, "{}", value) {
    //         //     Err(why) => { panic!("couldn't write to {}: {}", name, why.description()) }
    //         //     Ok(_) => println!("successfully wrote to {}", name),
    //         // }
    //     }
    //     writeln!(outfile, "").expect("could not write to file")
    // }
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