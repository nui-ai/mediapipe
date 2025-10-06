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

fn read_reference_data(filename: &str) -> anyhow::Result<Vec<PipelineOutputData>> {
    use protobuf::{CodedInputStream, error::{WireError, ProtobufError}};
    let mut file = File::open(filename)
        .with_context(|| format!("Failed to open reference proto file: {}", filename))?;
    let mut cis = CodedInputStream::new(&mut file);
    let mut out = Vec::new();
    loop {
        match cis.read_message::<PipelineOutputData>() {
            Ok(msg) => {
                // If the message is empty, we've reached EOF or a zero-length message
                if msg == PipelineOutputData::new() {
                    break;
                }
                out.push(msg);
            }
            Err(ProtobufError::WireError(WireError::UnexpectedEof)) => {
                break;
            }
            Err(e) => {
                if out.is_empty() {
                    anyhow::bail!("Failed to parse any messages from {}: {}", filename, e);
                } else {
                    break;
                }
            }
        }
    }
    println!("Loaded {} reference records from {}", out.len(), filename);
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Args: {:?}", args);

    // Read graph file as string, then convert to bytes
    let graph_string = std::fs::read_to_string(&args.graph_file)
        .with_context(|| format!("Failed to open graph file as string: {}", args.graph_file.display()))?;

    // Print the first 100 chars before moving the string
    println!("First 100 chars of graph file as string: {}", &graph_string[..std::cmp::min(100, graph_string.len())]);

    // Create a CString for safer FFI transfer
    let graph_cstring = CString::new(graph_string.clone())
        .with_context(|| "Failed to convert graph file to CString (found null byte?)")?;
    let graph_bytes = graph_cstring.as_bytes();

    println!("Loaded graph file: {} ({} bytes)", args.graph_file.display(), graph_bytes.len());
    println!("First 100 bytes of graph file as CString: {:?}", &graph_bytes[..std::cmp::min(100, graph_bytes.len())]);

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

    // Read reference proto file
    let reference_proto_path = "../output_data_v0.10.13.pb";
    let reference_data = read_reference_data(reference_proto_path)?;

    // Create pipeline operator via FFI
    let output_streams_csv = CString::new("multi_hand_landmarks,multi_hand_world_landmarks,multi_handedness")?;
    let handle = unsafe {
        ffi::hands_pipeline_operator_create(
            graph_cstring.as_ptr(),
            graph_string.len(), // Pass original string length without null terminator
            output_streams_csv.as_ptr(),
        )
    };
    if handle.is_null() {
        let err = unsafe { CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
        anyhow::bail!("Failed to create HandsPipelineOperator via C API: {}", err);
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
