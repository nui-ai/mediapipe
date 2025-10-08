use clap::Parser;
use ffmpeg_next as ffmpeg;
use std::ffi::{CStr, CString};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use anyhow::Context;

mod ffi;

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


fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Args: {:?}", args);

    ffmpeg::init()?;
    let input_path = args.input_video_path.as_ref().context("input_video_path is required")?;
    if !input_path.exists() {
        eprintln!("Input video file does not exist: {:?}", input_path);
        std::process::exit(1);
    }
    let mut ictx = ffmpeg::format::input(&input_path)?;
    let video_stream_index = ictx.streams().position(|s| s.parameters().medium() == ffmpeg::media::Type::Video);
    match video_stream_index {
        Some(idx) => {},
        None => {
            eprintln!("No video stream found in file: {:?}", input_path);
            std::process::exit(1);
        }
    }
    let video_stream_index = video_stream_index.unwrap();
    let stream = ictx.streams().nth(video_stream_index).unwrap();
    let codec_params = stream.parameters();
    let codec_ctx = ffmpeg::codec::context::Context::from_parameters(codec_params)
        .context("Failed to create codec context from parameters")?;
    let mut decoder = codec_ctx.decoder().video()
        .context("Failed to open video decoder")?;
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )?;
    println!("Scaler initialized for RGB24 conversion");

    let graph_file_cstr = CString::new(args.graph_file.to_str().unwrap())?;
    let output_streams_csv_cstr = CString::new("multi_hand_landmarks,multi_hand_world_landmarks,multi_handedness")?;
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


    println!("Preparing to process frames...");
    let mut frame_index = 0;
    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }
        if let Err(_e) = decoder.send_packet(&packet) {
            continue;
        }
        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match decoder.receive_frame(&mut decoded) {
                Ok(_) => {
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;
                    let data = rgb_frame.data(0);
                    let width = rgb_frame.width();
                    let height = rgb_frame.height();
                    let channels = 3; // RGB24
                    let timestamp_us = frame_index * 40_000; // ~25fps
                    let push_status = unsafe {
                        ffi::hands_pipeline_operator_push_image(
                            handle,
                            data.as_ptr(),
                            width as i32,
                            height as i32,
                            channels as i32,
                            timestamp_us,
                        )
                    };
                    if push_status != 0 {
                        let err = unsafe { CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
                        anyhow::bail!("push_image failed: {}", err);
                    }
                    let mut output_data: *mut i8 = std::ptr::null_mut();
                    let mut output_size: usize = 0;
                    let wait_status = unsafe {
                        ffi::hands_pipeline_operator_wait_for_output(
                            handle,
                            frame_index as i32,
                            &mut output_data,
                            &mut output_size,
                        )
                    };
                    if wait_status != 0 {
                        let err = unsafe { CStr::from_ptr(ffi::hands_pipeline_operator_get_last_error()) }.to_string_lossy();
                        anyhow::bail!("wait_for_output failed: {}", err);
                    }
                    let output_slice = unsafe { std::slice::from_raw_parts(output_data as *const u8, output_size) };
                    frame_index += 1;
                    if frame_index >= 999_999 {
                        break;
                    }
                }
                Err(err) => {
                    let err_str = err.to_string();
                    if err_str.contains("Resource temporarily unavailable") || err_str.contains("EAGAIN") {
                        break;
                    }
                    continue;
                }
            }
        }
    }
    unsafe { ffi::hands_pipeline_operator_destroy(handle); }
    Ok(())
}
