use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {

    // 1. generate rust code for being able to actually use the output protobuf types which the pipeline output is,
    //    from rust. reliant on the protobuf-codegen crate.
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

    protobuf_codegen::Codegen::new()
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

    // 2. Dynamically link the C API shared library for hand tracking by its path.
    //    This tells Cargo/rustc to add the Bazel output directory to the linker search path,
    //    so it can find the shared library built by Bazel from the C API, libhands_pipeline_operator_c_api.so, at build time.
    //    The library name (hands_pipeline_operator_c_api) must match the .so filename (libhands_pipeline_operator_c_api.so).
    //    At runtime, you don't also need to set LD_LIBRARY_PATH to include this directory so the dynamic loader
    //    can find the .so file.

    let so_dir = format!("{}/../bazel-bin/mediapipe/examples/desktop/hand_tracking", env!("CARGO_MANIFEST_DIR"));
    // direct rustc to find the static lib file (.a) during its compilation and linking when building the rust binary
    println!("cargo:rustc-link-search=native={}", so_dir);
    println!("cargo:rustc-link-lib=static=hands_pipeline_operator_c_api_static");
}
