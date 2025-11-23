/// FFI bindings for our mediapipe pipeline driving C API
/// that C API caters for using a mediapipe pipeline,
/// or at least the one we use in its final minified form
/// where the entire pipeline has been reduced to just
/// a single calculator.

extern "C" {
    pub fn hands_pipeline_operator_create(
        max_num_hands: u32,
        graph_file_path: *const libc::c_char,
        output_streams_csv: *const libc::c_char,
    ) -> *mut std::ffi::c_void;

    pub fn hands_pipeline_operator_push_image(
        handle: *mut std::ffi::c_void,
        image_data: *const u8,
        cols: i32,
        rows: i32,
        channels: i32,
        timestamp_us: usize,
    ) -> i32;

    pub fn hands_pipeline_operator_wait_for_output(
        handle: *mut std::ffi::c_void,
        frame_index: i32,
        output_data: *mut *mut libc::c_char,
        output_size: *mut usize,
    ) -> i32;

    pub fn hands_pipeline_operator_finalize(handle: *mut std::ffi::c_void) -> i32;
    pub fn hands_pipeline_operator_get_last_error() -> *const libc::c_char;
}
