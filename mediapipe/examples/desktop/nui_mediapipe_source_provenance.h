// Compile-time source provenance shared by the nui.ai hand and face C APIs.
//
// A compiled C++ library cannot ask Git which source revision produced it:
// after compilation it has neither a source checkout nor necessarily a Git
// executable. The build process therefore supplies that information to the
// C++ compiler. Core's build.rs uses Bazel's `--per_file_copt` option to pass
// definitions of the following form only while compiling the hand and face C
// API source files:
//
//   -DNUI_AI_MEDIAPIPE_SOURCE_COMMIT=<full Git object name>
//   -DNUI_AI_MEDIAPIPE_SOURCE_DIRTY=0
//   -DNUI_AI_MEDIAPIPE_BUILT_BY_CORE_BUILD_RS=1
//
// `--per_file_copt=<source filter>@<compiler option>` tells Bazel which source
// files should receive an option, and Bazel forwards the value after `@` to
// their C++ compiler commands. The `-D` syntax makes the named value available
// to the C++ preprocessor exactly as though a `#define` with that name and
// value appeared at the top of the source file. The accessors below turn those
// definitions into ordinary C++ string/integer constants. Compilation places
// the string bytes and accessor machine code in an object file, and the final
// link consequently places them inside libhand_tracking_c_api.so or
// libface_tracking_c_api.so. Scoping the options to these two source files also
// prevents a changed commit value from recompiling unrelated MediaPipe and
// OpenCV sources.
//
// Nothing is read from Git, the environment, or the filesystem when a client
// later asks the loaded library for these values. The answers physically
// belong to the particular shared-library artifact which answered the call.
// A direct Bazel build which omits the definitions remains supported: it is
// explicitly distinguishable by an unavailable revision/dirty state and a
// false "built by Core's build.rs" value.

#ifndef MEDIAPIPE_EXAMPLES_DESKTOP_NUI_MEDIAPIPE_SOURCE_PROVENANCE_H_
#define MEDIAPIPE_EXAMPLES_DESKTOP_NUI_MEDIAPIPE_SOURCE_PROVENANCE_H_

namespace nui_ai_mediapipe_source_provenance {

// Two macro levels are needed here. The outer macro expands the commit macro
// to its hexadecimal value; the inner macro then places quotes around that
// expanded value so it becomes a C++ string literal compiled into the library.
#define NUI_AI_MEDIAPIPE_STRINGIFY_INNER(value) #value
#define NUI_AI_MEDIAPIPE_STRINGIFY(value) \
  NUI_AI_MEDIAPIPE_STRINGIFY_INNER(value)

static inline const char* SourceCommit() {
#ifdef NUI_AI_MEDIAPIPE_SOURCE_COMMIT
  return NUI_AI_MEDIAPIPE_STRINGIFY(NUI_AI_MEDIAPIPE_SOURCE_COMMIT);
#else
  return "unavailable (not supplied by the build invocation)";
#endif
}

// Returns 0 for a clean source tree, 1 for a dirty source tree, and -1 when
// the build invocation did not supply the state. Keeping "unknown" distinct
// from "clean" prevents a direct Bazel build from making a false claim.
static inline int SourceDirty() {
#ifdef NUI_AI_MEDIAPIPE_SOURCE_DIRTY
#if NUI_AI_MEDIAPIPE_SOURCE_DIRTY != 0 && NUI_AI_MEDIAPIPE_SOURCE_DIRTY != 1
#error "NUI_AI_MEDIAPIPE_SOURCE_DIRTY must be 0 or 1"
#endif
  return NUI_AI_MEDIAPIPE_SOURCE_DIRTY;
#else
  return -1;
#endif
}

// Core defines this marker only on the Bazel command which its build.rs
// launches. Merely building the same target directly with Bazel omits the
// definition and therefore produces an artifact which truthfully returns 0.
static inline int LibraryBuiltByCoreBuildRs() {
#ifdef NUI_AI_MEDIAPIPE_BUILT_BY_CORE_BUILD_RS
#if NUI_AI_MEDIAPIPE_BUILT_BY_CORE_BUILD_RS != 1
#error "NUI_AI_MEDIAPIPE_BUILT_BY_CORE_BUILD_RS must be 1 when supplied"
#endif
  return 1;
#else
  return 0;
#endif
}

#undef NUI_AI_MEDIAPIPE_STRINGIFY
#undef NUI_AI_MEDIAPIPE_STRINGIFY_INNER

}  // namespace nui_ai_mediapipe_source_provenance

#endif  // MEDIAPIPE_EXAMPLES_DESKTOP_NUI_MEDIAPIPE_SOURCE_PROVENANCE_H_
