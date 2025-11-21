/// FFI bindings for our no-pipeline hand tracking C API:
///
/// • a result struct HandTrackingResultC is mirrored here to bind to the HandTrackingResult result type of the C API;
///   this struct holds the result of the hand tracking process function, which is a nested structure consisting of
///   the object landmarks, viewport landmarks, and handedness classifications per hand. so the definitions herein
///   begin with those sub-structs ffi definitions and ends with the top-level HandTrackingResultC structure
///   definition which uses them.
/// • the C API functions' FFI bindings follow theses struct definitions at the bottom.

#[repr(C)]
pub struct LandmarkC {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub visibility: f32,
    pub presence: f32,
}

pub const NUM_LANDMARKS: usize = 21;

#[repr(C)]
pub struct LandmarkListC {
    pub landmark: [LandmarkC; NUM_LANDMARKS],
}

#[repr(C)]
pub struct NormalizedLandmarkC {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub visibility: f32,
    pub presence: f32,
}

#[repr(C)]
pub struct NormalizedLandmarkListC {
    pub landmark: [NormalizedLandmarkC; NUM_LANDMARKS],
}

#[repr(C)]
pub struct ClassificationC {
    pub index: i32,
    pub score: f32,
    pub label: *const libc::c_char,
    pub display_name: *const libc::c_char,
}

#[repr(C)]
pub struct ClassificationListC {
    pub classification: *mut ClassificationC,
    pub classification_count: libc::size_t,
}

#[repr(C)]
pub struct HandTrackingResultC {
    pub normalized_landmarks: *mut NormalizedLandmarkListC,
    pub landmarks: *mut LandmarkListC,
    pub classifications: *mut ClassificationListC,
}

/// C API functions
extern "C" {
    pub fn hand_tracking_core_create(max_num_hands: u32) -> *mut std::ffi::c_void;

    /// function for processing a single image frame, returning a pointer to a HandTrackingResultC struct
    /// in the provided output argument `result_out`. after each call to this function, the caller is responsible
    /// for managing the lifecycle of the HandTrackingResultC struct's elements returned as that pointer, by eventually
    /// calling the below convenience function `hand_tracking_result_destroy` on it at some point for each such struct
    /// returned (each call returns one such struct), otherwise the structs accumulate in memory. or the caller may
    /// individually destroy each of the struct's constituents in separation, as long as they are all eventually
    /// destroyed (in place of using the below convenience function which destroys the entire struct's payload).
    pub fn hand_tracking_core_process(
        opaque_handle: *mut std::ffi::c_void,
        data: *const u8,
        width: libc::size_t,
        height: libc::size_t,
        row_stride: libc::size_t,
        hand_tracking_result: *mut *mut HandTrackingResultC, // output argument
    ) -> i32;

    /// convenience function for destroying a HandTrackingResultC with all its nested allocations,
    /// to be called by the caller for each HandTrackingResultC returned from each call to `hand_tracking_core_process`.
    /// In the future, we may switch to efficient memory management of the result of `hand_tracking_core_process` by:
    /// 1. preallocating a fixed amount of memory and reuse the same memory on all calls to the underlying method
    /// 2. passing this result data by reference.
    /// ― at which point the caller will no longer need to manage the lifecycle (destruction) of the returned result
    /// from `hand_tracking_core_process` on the caller side, but rather as part of the finalization API function
    /// (hand_tracking_core_finalize).
    pub fn hand_tracking_result_destroy(result: *mut HandTrackingResultC);

    pub fn hand_tracking_core_finalize(handle: *mut std::ffi::c_void) -> i32;
    pub fn hand_tracking_get_last_error() -> *const libc::c_char;
}
