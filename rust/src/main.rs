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

mod ffi;
mod proto;
use proto::pipeline_output::PipelineOutputData;

extern "C" {
    fn hands_pipeline_operator_init_protobuf();
    fn hands_pipeline_operator_force_link_protos();
}

fn init_protos() {
    println!("Initializing protobuf library");
    unsafe { hands_pipeline_operator_init_protobuf(); }
    println!("Protobuf library initialized, registering descriptors");
    unsafe { hands_pipeline_operator_force_link_protos(); }
    println!("Protobuf descriptors registered");
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    #[arg(long)]
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
        if len == 0 { continue; } // skip zero-length
        let mut msg_buf = vec![0u8; len as usize];
        reader.read_exact(&mut msg_buf)?;
        let msg = T::parse_from_bytes(&msg_buf)?;
        out.push(msg);
    }
    Ok(out)
}

fn read_reference_data(filename: &str) -> anyhow::Result<Vec<PipelineOutputData>> {
    use std::io::BufReader;
    let file = File::open(filename)
        .with_context(|| format!("Failed to open reference proto file: {}", filename))?;
    let mut reader = BufReader::new(file);
    let out = read_delimited_protobuf_messages::<PipelineOutputData>(&mut reader)?;
    println!("Loaded {} reference records from {}", out.len(), filename);
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Args: {:?}", args);

    // Ensure protobuf descriptors are registered, which our C++ main using the same api doesn't do,
    // nor does it avoid crashing over protobuf initialization in the case of the current rust main.
    // it's not causing it either (comment it out and we fail the same).
    // init_protos();

    // Open video/camera
    let mut capture = if let Some(ref path) = args.input_video_path {
        VideoCapture::from_file(path.to_str().unwrap(), CAP_ANY)?
    } else {
        VideoCapture::new(0, CAP_ANY)?
    };
    if !capture.is_opened()? {
        anyhow::bail!("Failed to open video/camera: {:?}", args.input_video_path);
    }
    println!("Video/camera opened successfully");

    // Check if reference proto file exists and is readable
    let reference_proto_path = "output_data_v0.10.13.pb";
    if !std::path::Path::new(reference_proto_path).is_file() {
        eprintln!("Error: Reference proto file '{}' not found or not a regular file.", reference_proto_path);
        std::process::exit(1);
    }
    if std::fs::metadata(reference_proto_path).map(|m| m.len()).unwrap_or(0) == 0 {
        eprintln!("Error: Reference proto file '{}' is empty.", reference_proto_path);
        std::process::exit(1);
    }
    println!("Reference proto file '{}' found and is non-empty.", reference_proto_path);
    let reference_data = read_reference_data(reference_proto_path)?;

    // print the current working directory
    let current_dir = std::env::current_dir()?;
    println!("Current working directory: {}", current_dir.display());

    // Create pipeline operator via FFI
    let graph_file_cstr = CString::new(args.graph_file.to_str().unwrap())?;
    let output_streams_csv_cstr = CString::new("multi_hand_landmarks,multi_hand_world_landmarks,multi_handedness")?;

    // explicitly check graph_file_cstr.as_ptr() is not a nullptr
    if graph_file_cstr.as_ptr().is_null() {
        eprintln!("Error: graph_file_cstr is a null pointer");
        std::process::exit(1);
    }
    let handle = unsafe {
        ffi::hands_pipeline_operator_create(
            graph_file_cstr.as_ptr(),
            output_streams_csv_cstr.as_ptr(),
        )
    };
    if handle.is_null() {
        let err = unsafe { std::ffi::CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) };
        eprintln!("Error: Failed to create HandsPipelineOperator via C API: {}", err.to_string_lossy());
        std::process::exit(1);
    }
    println!("Pipeline operator created successfully");

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
                eprintln!("Empty frame from camera being ignored");
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
            ffi::hands_pipeline_operator_push_image(
                handle,
                input_frame.data(),
                input_frame.cols(),
                input_frame.rows(),
                input_frame.channels(),
                timestamp_us,
            )
        };
        if push_status != 0 {
            let err = unsafe { CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
            anyhow::bail!("push_image failed: {}", err);
        }

        // Wait for output
        let mut output_data: *mut i8 = std::ptr::null_mut();
        let mut output_size: usize = 0;
        let wait_status = unsafe {
            ffi::hands_pipeline_operator_wait_for_output(
                handle,
                i as i32,
                &mut output_data,
                &mut output_size,
            )
        };
        if wait_status != 0 {
            let err = unsafe { CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
            anyhow::bail!("wait_for_output failed: {}", err);
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
                    eprintln!("Pipeline output at frame {} is different than the reference output.", i);
                    eprintln!("Terminating early due to difference in output at frame {}", i);
                    break;
                } else {
                    println!("Pipeline output for frame {} is identical to its reference output.", i);
                }
            } else {
                eprintln!("Reference output file doesn't have data for frame {}", i);
            }
        }
    }
    output_proto_file.flush()?;
    let finalize_status = unsafe { ffi::hands_pipeline_operator_finalize(handle) };
    if finalize_status != 0 {
        let err = unsafe { CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
        eprintln!("Error during mediapipe graph finalization: {}", err);
    }
    unsafe { ffi::hands_pipeline_operator_destroy(handle); }
    Ok(())
}
