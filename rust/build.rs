use std::env;
use std::fs;
use std::path::PathBuf;

/// custom build script, idiomatically pushing build commands to cargo via println!()
/// this build script carries out the following build functions:
///
///   - generates rust sources from protobuf files of mediapipe pipeline output types of the hands pipeline which we reuse across languages for convenience
///   - dynamically links our C API for hand tracking (built by Bazel from the current repository's C++ codebase) ― the pipeline based one
///   - dynamically links our C API for hand tracking (built by Bazel from the current repository's C++ codebase) ― the no-pipeline based one
///   - dynamically links our vendored version of OpenCV expected by (and verified to work for) our C API for hand tracking
///
/// these build functions cannot (as far as we know) be carried out by plain cargo definitions,
/// thus requiring use of a rust-idiomatic use of a custom build main like the current one here.
/// this build script is invoked by cargo at build time, before any (other) rust code is compiled.

fn main() {

    eprintln!("custom build script build.rs is starting");

    // Step 1:
    // generate rust code for being able to actually use the output protobuf types which are the output types of our C pipeline API for hand tracking, reliant on the protobuf-codegen crate.
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

    // Step 2:
    // Dynamically link the OpenCV shared libraries that we bazel build, which we transitively depend on via the said C API,
    // this is in order to avoid the system OpenCV being linked which would cause https://github.com/nui-ai/mediapipe/issues/22.
    // our .cargo/config.toml only handles this aspect for the build of the OpenCV crate, whereas this here handles
    // the same loading for our final rust binary, which must also link against that bazel-built OpenCV of ours
    // and not the system one, at runtime.
    let opencv_dir = format!("{}/../bazel-bin/third_party/opencv_cmake/lib", env!("CARGO_MANIFEST_DIR"));  // we don't use the opencv_cmake_for_rust copy of them here, but the versioned copy.
    eprintln!("cargo will try loading OpenCV from: {}", opencv_dir);
    println!("cargo:rustc-link-arg=-Wl,--enable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", opencv_dir);

    // Step 3:
    // Dynamically link our shared library housing our said C API for hand tracking ― by its path.
    // This tells Cargo/rustc to add the Bazel output directory to the linker search path,
    // so it can find the shared library built by Bazel from the C API, libhands_pipeline_operator_c_api.so, at build time.
    // The library name (hands_pipeline_operator_c_api) must match the .so filename (libhands_pipeline_operator_c_api.so).
    // At runtime, you don't also need to set LD_LIBRARY_PATH to include this directory so the dynamic loader
    // can find the .so file.
    //
    // Why dynamic linking?
    // static linking our C API library would make it necessary to have all dependencies of the C API as static libraries too,
    // and their own dependencies too as this requirement is recursively transitive, which is typically intractable or extremely
    // hard to accoplish in dependency chains which aren't very small, so dynamic linking the C API library is our only option.
    //
    // therefor, the object of integration providing the C library to rust is the bazel built library file of the C API,
    // and it is dynamically linked by the rust binary due to the above reason.
    //
    //
    // Theoretically we could alternatively directly feed in an object file from the Bazel build to the rust build instead of
    // using the bazel built library of the C API as the object of integration for rust to consume, but such C++ object files
    // may likely not play well in rust's mechanics of linking as that would be a non-standard mix. also in terms of build
    // lifecycle mechanics bazel does not in its default behavior leave behind object files for cargo to consume nor trigger
    // cargo once it is done building its part.

    let hand_tracking_so_dir = format!("{}/../bazel-bin/mediapipe/examples/desktop/hand_tracking", env!("CARGO_MANIFEST_DIR"));
    // direct rustc to find the lib file (.so) during compilation and linking when building the rust binary
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", hand_tracking_so_dir);
    // bake into the built binary that at runtime the same lib file will be found when the rust binary is being run;
    // the built binary will essentially hold a dynamic linking definition saying to load it from there when run:
    println!("cargo:rustc-link-search=native={}", hand_tracking_so_dir);

    // Step 3a: link our pipeline-backed C api for hand tracking
    println!("cargo:rustc-link-lib=dylib=hands_pipeline_operator_c_api");

    // Step 3b: link our no-pipeline-backed C api for hand tracking
    println!("cargo:rustc-link-lib=dylib=hand_tracking_c_api");
}
