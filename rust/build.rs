use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {

    eprintln!("custom build script build.rs is starting");

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

    // 2. Dynamically link the OpenCV shared libraries that we bazel build, which we transitively depend on via the C API,
    //    this is in order to avoid the system OpenCV being linked which would cause https://github.com/nui-ai/mediapipe/issues/22.
    //    our .cargo/config.toml only handles this aspect for the build of the OpenCV crate, whereas this here handles
    //    the same loading for our final rust binary, which must also link against that bazel-built OpenCV of ours
    //    and not the system one, at runtime.
    let opencv_dir = format!("{}/../bazel-bin/third_party/opencv_cmake/lib", env!("CARGO_MANIFEST_DIR"));  // we don't use the opencv_cmake_for_rust copy of them here, but the versioned copy.
    eprintln!("cargo will try loading OpenCV from: {}", opencv_dir);
    println!("cargo:rustc-link-arg=-Wl,--enable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", opencv_dir);

    // 2. Dynamically link the C API shared library for hand tracking by its path.
    //    This tells Cargo/rustc to add the Bazel output directory to the linker search path,
    //    so it can find the shared library built by Bazel from the C API, libhands_pipeline_operator_c_api.so, at build time.
    //    The library name (hands_pipeline_operator_c_api) must match the .so filename (libhands_pipeline_operator_c_api.so).
    //    At runtime, you don't also need to set LD_LIBRARY_PATH to include this directory so the dynamic loader
    //    can find the .so file.
    let so_dir = format!("{}/../bazel-bin/mediapipe/examples/desktop/hand_tracking", env!("CARGO_MANIFEST_DIR"));
    // direct rustc to find the lib file (.so) during its compilation and linking when building the rust binary
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", so_dir);
    // make the runtime find the same lib file when running the rust built binary
    println!("cargo:rustc-link-search=native={}", so_dir);

    // static linking our C API library would make it necessary to have all dependencies of the C API as static libraries too,
    // and their own dependencies too as this requirement is recursively transitive, which is typically intractable or extremely
    // hard to accoplish in dependency chains which aren't very small, so dynamic linking the C API library is our only option.
    //
    // theoretically we could alternatively directly feed in an object file from the Bazel build to the rust build instead of
    // using the bazel built library of the C API as the object of integration for rust to consume,
    // but such C++ object files may likely not play well in rust's mechanics of linking as that would be a non-standard mix.
    // also in terms of build lifecycle mechanics bazel does not in its default behavior leave behind object files for cargo
    // to consume nor trigger cargo once it is done building its part.
    //
    // therefor, the object of integration providing the C library to rust is the bazel built library file of the C API,
    // and it is dynamically linked by the rust binary due to the above reason.
    println!("cargo:rustc-link-lib=dylib=hands_pipeline_operator_c_api");
}
