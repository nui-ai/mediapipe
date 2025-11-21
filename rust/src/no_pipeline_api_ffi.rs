/// FFI bindings for our no-pipeline hand tracking C API

extern "C" {
    pub fn hand_tracking_core_create(max_num_hands: u32) -> *mut std::ffi::c_void;

    pub fn process_numpy_image_from_rust(
        opaque_handle: *mut std::ffi::c_void,
        data: *const u8,
        height: libc::size_t,
        width: libc::size_t,
        stride_row: libc::size_t,
        stride_col: libc::size_t,
    ) -> i32;

    // pub fn hands_pipeline_operator_wait_for_output(
    //     handle: *mut std::ffi::c_void,
    //     frame_index: i32,
    //     output_data: *mut *mut libc::c_char,
    //     output_size: *mut usize,
    // ) -> i32;

    pub fn hand_tracking_core_finalize(handle: *mut std::ffi::c_void) -> i32;
    pub fn hand_tracking_get_last_error() -> *const libc::c_char;
}
