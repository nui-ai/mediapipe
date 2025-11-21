use clap::Parser;
use opencv::core::Mat;
use opencv::imgproc;
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY};
use protobuf::Message; // Use rust-protobuf Message trait
use std::ffi::{CStr, CString};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write}; // removed Read
use std::path::PathBuf;
use anyhow::Context;

mod pipeline_api_ffi;
mod proto;

use proto::pipeline_output::PipelineOutputData;

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    #[arg(long, required=true, help = "maximum number of hands to track; we still need to explicitly provide this hand tracking argument")]
    pub max_num_hands: u32,
    #[arg(long, required=true)]
    pub graph_file: PathBuf,
    #[arg(long)]
    pub input_video_path: Option<PathBuf>,
    #[arg(long)]
    pub output_video_path: Option<PathBuf>,
}

// Helper to read length-delimited protobuf messages from a file.
// Note: The rust-protobuf library does not natively support reading length-delimited protobuf messages from a file.
// This helper implements the logic manually to match the output of C++ SerializeDelimitedToOstream.
fn read_delimited_protobuf_messages<T: protobuf::Message + Default>(reader: &mut std::io::BufReader<std::fs::File>) -> anyhow::Result<Vec<T>> {
    use std::io::Read;
    let mut out = Vec::new();
    loop {
        // Read varint length
        let mut len_buf = [0u8; 1];
        let mut len = 0u64;
        let mut shift = 0;
        let mut read_any = false;
        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {
                    read_any = true;
                    let byte = len_buf[0];
                    len |= ((byte & 0x7F) as u64) << shift;
                    if byte & 0x80 == 0 { break; }
                    shift += 7;
                }
                Err(_) => {
                    if !read_any { break; } // EOF
                    else { anyhow::bail!("Failed to read varint length"); }
                }
            }
        }
        if !read_any { break; } // EOF
        // Do not skip zero-length messages
        let mut msg_buf = vec![0u8; len as usize];
        reader.read_exact(&mut msg_buf)?;
        let msg = T::parse_from_bytes(&msg_buf)?;
        out.push(msg);
    }
    Ok(out)
}

fn read_reference_data(filename: &str) -> anyhow::Result<Vec<PipelineOutputData>> {
    use std::io::BufReader;
    let file = File::open(filename).with_context(|| format!("Failed to open reference proto file: {}", filename))?;
    let mut reader = BufReader::new(file);
    let out = read_delimited_protobuf_messages::<PipelineOutputData>(&mut reader)?;
    println!("{} reference output records read from {}", out.len(), filename);
    Ok(out)
}

fn print_pipeline_output_diff(a: &PipelineOutputData, b: &PipelineOutputData, frame_number: usize) {
    if a.frame_number != b.frame_number {
        eprintln!("frame_number differs: {} vs {}", a.frame_number, b.frame_number);
    }
    if a.multi_hand_world_landmarks.len() != b.multi_hand_world_landmarks.len() {
        eprintln!("multi_hand_world_landmarks length differs: {} vs {}", a.multi_hand_world_landmarks.len(), b.multi_hand_world_landmarks.len());
    }
    for (i, (l1, l2)) in a.multi_hand_world_landmarks.iter().zip(b.multi_hand_world_landmarks.iter()).enumerate() {
        if l1 != l2 {
            eprintln!("multi_hand_world_landmarks[{}] differs: {:?} vs {:?}", i, l1, l2);
        }
    }
    if a.multi_hand_landmarks.len() != b.multi_hand_landmarks.len() {
        eprintln!("multi_hand_landmarks length differs: {} vs {}", a.multi_hand_landmarks.len(), b.multi_hand_landmarks.len());
    }
    for (i, (l1, l2)) in a.multi_hand_landmarks.iter().zip(b.multi_hand_landmarks.iter()).enumerate() {
        if l1 != l2 {
            eprintln!("multi_hand_landmarks[{}] differs: {:?} vs {:?}", i, l1, l2);
        }
    }
    if a.multi_handedness.len() != b.multi_handedness.len() {
        eprintln!("multi_handedness length differs: {} vs {}", a.multi_handedness.len(), b.multi_handedness.len());
    }
    for (i, (c1, c2)) in a.multi_handedness.iter().zip(b.multi_handedness.iter()).enumerate() {
        if c1 != c2 {
            eprintln!("multi_handedness[{}] differs: {:?} vs {:?}", i, c1, c2);
        }
    }
    if a.hand_rects_from_palm_detections.len() != b.hand_rects_from_palm_detections.len() {
        eprintln!("hand_rects_from_palm_detections length differs: {} vs {}", a.hand_rects_from_palm_detections.len(), b.hand_rects_from_palm_detections.len());
    }
    for (i, (r1, r2)) in a.hand_rects_from_palm_detections.iter().zip(b.hand_rects_from_palm_detections.iter()).enumerate() {
        if r1 != r2 {
            eprintln!("hand_rects_from_palm_detections[{}] differs: {:?} vs {:?}", i, r1, r2);
        }
    }
}

fn cwdir() -> anyhow::Result<()> {
    let current_dir = std::env::current_dir().context("Failed to get current directory")?;
    let parent_dir = current_dir.parent().context("No parent directory")?;
    std::env::set_current_dir(parent_dir).context("Failed to change directory")?;
    println!("Changed working directory to: {}", std::env::current_dir()?.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {

    let args = Args::parse();
    println!("Args: {:?}", args);

    let current_dir = std::env::current_dir()?;
    println!("current working directory: {}", current_dir.display());

    // Open video/camera input
    let mut capture = if let Some(ref path) = args.input_video_path {
        VideoCapture::from_file(path.to_str().unwrap(), CAP_ANY)?
    } else {
        VideoCapture::new(0, CAP_ANY)?
    };
    if !capture.is_opened()? {
        anyhow::bail!("Failed to open video/camera: {:?}", args.input_video_path);
    }
    println!("Video/camera input successfully opened");

    // Check the reference output file
    let reference_proto_path = "../output_data_two_hands_num_hands_3_v0.10.13.pb";
    if !std::path::Path::new(reference_proto_path).is_file() {
        eprintln!("Error: the reference output data file '{}' not found or is not a regular file.", reference_proto_path);
        std::process::exit(1);
    }
    if std::fs::metadata(reference_proto_path).map(|m| m.len()).unwrap_or(0) == 0 {
        eprintln!("Error: the reference output data file '{}' is empty.", reference_proto_path);
        std::process::exit(1);
    }
    println!("reference output data file '{}' found and is non-empty.", reference_proto_path);
    let reference_data = read_reference_data(reference_proto_path)?;

    // Create the pipeline operator via FFI
    let graph_file_cstr = CString::new(args.graph_file.to_str().unwrap())?;
    let output_streams_csv_cstr = CString::new("multi_hand_landmarks,multi_hand_world_landmarks,multi_handedness")?;

    // temporarily change the working directory to the project-wide project root directory,
    // so that relative paths to pbtxt files expected by our mediapipe pipeline are resolved correctly,
    // as the mediapipe pipeline's loading of the pbtxt and other resource files can only search from a fixed base path ―
    // and it does not take a variable for its base path for model loading from rust, while our rust runs under our `rust`
    // directory, not the project-wide root directory (only because it's convenient in RustRover).
    cwdir().expect("failed setting the working directory for the mediapipe current pipeline");
    if graph_file_cstr.as_ptr().is_null() {
        eprintln!("Error: graph_file_cstr is a null pointer");
        std::process::exit(1);
    }
    let handle = unsafe {
        pipeline_api_ffi::hands_pipeline_operator_create(
            args.max_num_hands,
            graph_file_cstr.as_ptr(),
            output_streams_csv_cstr.as_ptr(),
        )
    };
    if handle.is_null() {
        let err = unsafe { std::ffi::CStr::from_ptr(pipeline_api_ffi::hands_pipeline_operator_get_last_error()) };
        eprintln!("Error: Failed to create HandsPipelineOperator via its C API: {}", err.to_string_lossy());
        std::process::exit(1);
    }
    println!("pipeline operator object created successfully");

    // Prepare output proto file
    let output_proto_path = "output_data_cpp.pb";
    let mut output_proto_file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(output_proto_path)
            .with_context(|| format!("Failed to open output proto file: {}", output_proto_path))?
    );

    let video_file_input = args.input_video_path.is_some();
    for i in 0..999_999 {
        let mut input_frame_raw = Mat::default();
        capture.read(&mut input_frame_raw)?;
        if input_frame_raw.empty() {
            if !video_file_input {
                eprintln!("empty frame from camera being ignored");
                continue;
            }
            break;
        }

        // Convert to RGB and flip if needed
        let mut input_frame = Mat::default();
        imgproc::cvt_color(&input_frame_raw, &mut input_frame, imgproc::COLOR_BGR2RGB, 0)?;
        if !video_file_input {
            let mut flipped = Mat::default();
            opencv::core::flip(&input_frame, &mut flipped, 1)?;
            input_frame = flipped;
        }

        // Push image to pipeline
        let tick_count = opencv::core::get_tick_count()?;
        let tick_freq = opencv::core::get_tick_frequency()?;
        let timestamp_us = (tick_count as f64 / tick_freq * 1e6) as usize;
        let push_status = unsafe {
            pipeline_api_ffi::hands_pipeline_operator_push_image(
                handle,
                input_frame.data(),
                input_frame.cols(),
                input_frame.rows(),
                input_frame.channels(),
                timestamp_us,
            )
        };
        if push_status != 0 {
            let err = unsafe { CStr::from_ptr(pipeline_api_ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
            anyhow::bail!("pushing an image to the pipeline failed: {}", err);
        }

        // Wait for output
        let mut output_data: *mut i8 = std::ptr::null_mut();
        let mut output_size: usize = 0;
        let wait_status = unsafe {
            pipeline_api_ffi::hands_pipeline_operator_wait_for_output(
                handle,
                i as i32,
                &mut output_data,
                &mut output_size,
            )
        };
        if wait_status != 0 {
            let err = unsafe { CStr::from_ptr(pipeline_api_ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
            anyhow::bail!("waiting for pipeline output failed: {}", err);
        }

        // Parse output proto
        let output_slice = unsafe { std::slice::from_raw_parts(output_data as *const u8, output_size) };
        let stream_data_msg = PipelineOutputData::parse_from_bytes(output_slice)?;
        unsafe { libc::free(output_data as *mut libc::c_void); }

        // Write to file (delimited)
        stream_data_msg.write_length_delimited_to_writer(&mut output_proto_file)?;

        // Compare with reference data if available
        if !reference_data.is_empty() {
            if i < reference_data.len() {
                if stream_data_msg != reference_data[i] {
                    eprintln!("pipeline output at frame {} is different than the reference output:", i);
                    print_pipeline_output_diff(&stream_data_msg, &reference_data[i], i);
                    eprintln!("terminating early due to difference in output at frame {}", i);
                    break;
                } else {
                    println!("pipeline output for frame {} is identical to its reference output.", i);
                }
            } else {
                eprintln!("reference output file doesn't have data for frame {}", i);
            }
        }
    }
    output_proto_file.flush()?;
    let finalize_status = unsafe { pipeline_api_ffi::hands_pipeline_operator_finalize(handle) };
    if finalize_status != 0 {
        let err = unsafe { CStr::from_ptr(pipeline_api_ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
        eprintln!("Error during mediapipe graph finalization: {}", err);
    }
    Ok(())
}
