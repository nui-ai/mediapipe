use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let proto_files = [
        "../mediapipe/examples/desktop/pipeline_output.proto",
        "../mediapipe/framework/formats/landmark.proto",
        "../mediapipe/framework/formats/classification.proto",
        "../mediapipe/framework/formats/rect.proto",
    ];
    let proto_files: Vec<_> = proto_files
        .iter()
        .map(|f| PathBuf::from(&manifest_dir).join(f))
        .collect();
    let out_dir = PathBuf::from(&manifest_dir).join("src/proto");
    fs::create_dir_all(&out_dir).unwrap();

    protobuf_codegen_pure::Codegen::new()
        .out_dir(&out_dir)
        .inputs(
            &proto_files
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
        )
        .include(PathBuf::from(&manifest_dir).join("../"))
        .run()
        .expect("protoc codegen failed");

    // Link the C API shared library for hand tracking.
    // This tells Cargo/rustc to add the Bazel output directory to the linker search path,
    // so it can find the shared library built from the C API, libhands_pipeline_operator_c_api.so, at build time.
    // The library name (hands_pipeline_operator_c_api) must match the .so filename (libhands_pipeline_operator_c_api.so).
    // At runtime, you must also set LD_LIBRARY_PATH to this directory so the dynamic loader can find the .so file.
    // Example:
    //   export LD_LIBRARY_PATH=$PWD/../bazel-bin/mediapipe/examples/desktop/hand_tracking:$LD_LIBRARY_PATH
    // No additional code is needed in Rust to load the library; the extern "C" declarations in ffi.rs are sufficient.
    println!("cargo:rustc-link-search=native={}/../bazel-bin/mediapipe/examples/desktop/hand_tracking", manifest_dir);
    println!("cargo:rustc-link-lib=dylib=hands_pipeline_operator_c_api");
}
