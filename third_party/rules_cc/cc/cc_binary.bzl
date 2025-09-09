# This is a stub for @rules_cc//cc:cc_binary.bzl to satisfy dependencies that expect this file to exist.
# It is safe to ignore this file unless you see a build error from the fail() below.
# If you see the error, a dependency is trying to use the cc_binary rule, which is not supported in this build.
def _cc_binary_impl(ctx):
    fail("This is a stub for @rules_cc//cc:cc_binary.bzl for compatibility. If you see this error, a dependency is trying to use the cc_binary rule, which is not supported in this build. If you do not see this error, you are safe to ignore this file.")

cc_binary = rule(
    implementation = _cc_binary_impl,
    attrs = {},
)
