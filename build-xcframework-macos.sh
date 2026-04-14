#!/bin/bash
#
# Build a macOS-only whisper.xcframework.
#
# Motivation: the stock build-xcframework.sh uses cmake's Xcode generator,
# which on nix-darwin + Xcode 26 hits two problems:
#
#   1. `xcode-select -p` points at a nix-provided Apple SDK 14.4 stub rather
#      than the real Xcode, so cmake resolves the wrong sysroot.
#   2. Even with DEVELOPER_DIR forced to the real Xcode, the cmake 4 +
#      Xcode 26 combo emits link commands inside the generated project that
#      invoke `ld` directly with clang-driver flags (-Xlinker, -Wl,...),
#      which the linker then rejects as "unknown options".
#
# This script sidesteps both by using the Unix Makefiles generator (cmake
# drives clang directly, no Xcode project in the loop) and assembling the
# framework / xcframework manually. VoiceInk is macOS-only, so skipping iOS /
# visionOS / tvOS slices is fine.
#
# Output: build-apple/whisper.xcframework

set -euo pipefail

MACOS_MIN_OS_VERSION=${MACOS_MIN_OS_VERSION:-13.3}
JOBS=${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}

FRAMEWORK_NAME="whisper"
BUILD_DIR="build-macos"
OUT_DIR="build-apple"

# Force real Xcode.app — nix-darwin typically points xcode-select at a stub
REAL_XCODE="${REAL_XCODE:-/Applications/Xcode.app/Contents/Developer}"
if [[ -d "${REAL_XCODE}" ]]; then
    export DEVELOPER_DIR="${REAL_XCODE}"
fi
echo "Using DEVELOPER_DIR=${DEVELOPER_DIR:-<default>}"

MACOS_SDK=$(xcrun --sdk macosx --show-sdk-path)
XCODE_CLANG=$(xcrun -find clang)
XCODE_CLANGXX=$(xcrun -find clang++)
echo "Using macOS SDK: ${MACOS_SDK}"
echo "Using clang:     ${XCODE_CLANG}"
echo "Using clang++:   ${XCODE_CLANGXX}"

echo "==> Cleaning previous build"
rm -rf "${BUILD_DIR}" "${OUT_DIR}"

# Force cmake to use real Xcode's clang. On nix-darwin, cmake otherwise picks
# up the nix-wrapped clang from PATH, which doesn't know how to resolve the
# Xcode SDK's system headers (missing stdint, etc).
echo "==> Configuring (cmake, Makefiles generator)"
cmake -B "${BUILD_DIR}" -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER="${XCODE_CLANG}" \
    -DCMAKE_CXX_COMPILER="${XCODE_CLANGXX}" \
    -DCMAKE_OBJC_COMPILER="${XCODE_CLANG}" \
    -DCMAKE_OBJCXX_COMPILER="${XCODE_CLANGXX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOS_MIN_OS_VERSION}" \
    -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_OSX_SYSROOT="${MACOS_SDK}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DWHISPER_COREML=ON \
    -DWHISPER_COREML_ALLOW_FALLBACK=ON \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_METAL_USE_BF16=ON \
    -DGGML_BLAS_DEFAULT=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_NATIVE=OFF \
    -S .

echo "==> Building (j${JOBS})"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo "==> Combining static libraries"
LIBS=(
    "${BUILD_DIR}/src/libwhisper.a"
    "${BUILD_DIR}/src/libwhisper.coreml.a"
    "${BUILD_DIR}/ggml/src/libggml.a"
    "${BUILD_DIR}/ggml/src/libggml-base.a"
    "${BUILD_DIR}/ggml/src/libggml-cpu.a"
    "${BUILD_DIR}/ggml/src/ggml-metal/libggml-metal.a"
    "${BUILD_DIR}/ggml/src/ggml-blas/libggml-blas.a"
)
for lib in "${LIBS[@]}"; do
    if [[ ! -f "${lib}" ]]; then
        echo "missing expected static library: ${lib}" >&2
        exit 1
    fi
done

TMP_DIR="${BUILD_DIR}/tmp"
mkdir -p "${TMP_DIR}"
# Filter the benign "same member name (foo.o) in output file used for input
# files" noise libtool emits when combining static archives with overlapping
# object names. Real errors (missing objects, arch mismatches) are preserved.
libtool -static -o "${TMP_DIR}/combined.a" "${LIBS[@]}" 2> >(grep -v 'same member name' >&2)

echo "==> Assembling framework bundle"
FW_ROOT="${BUILD_DIR}/framework/${FRAMEWORK_NAME}.framework"
FW_VERSIONED="${FW_ROOT}/Versions/A"
rm -rf "${FW_ROOT}"
mkdir -p "${FW_VERSIONED}/Headers" "${FW_VERSIONED}/Modules" "${FW_VERSIONED}/Resources"

# macOS framework symlink layout
ln -sf A "${FW_ROOT}/Versions/Current"
ln -sf Versions/Current/Headers "${FW_ROOT}/Headers"
ln -sf Versions/Current/Modules "${FW_ROOT}/Modules"
ln -sf Versions/Current/Resources "${FW_ROOT}/Resources"
ln -sf "Versions/Current/${FRAMEWORK_NAME}" "${FW_ROOT}/${FRAMEWORK_NAME}"

# Headers
cp include/whisper.h            "${FW_VERSIONED}/Headers/"
cp ggml/include/ggml.h          "${FW_VERSIONED}/Headers/"
cp ggml/include/ggml-alloc.h    "${FW_VERSIONED}/Headers/"
cp ggml/include/ggml-backend.h  "${FW_VERSIONED}/Headers/"
cp ggml/include/ggml-metal.h    "${FW_VERSIONED}/Headers/"
cp ggml/include/ggml-cpu.h      "${FW_VERSIONED}/Headers/"
cp ggml/include/ggml-blas.h     "${FW_VERSIONED}/Headers/"
cp ggml/include/gguf.h          "${FW_VERSIONED}/Headers/"

cat > "${FW_VERSIONED}/Modules/module.modulemap" << 'EOF'
framework module whisper {
    header "whisper.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-blas.h"
    header "gguf.h"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

cat > "${FW_VERSIONED}/Resources/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>org.ggml.${FRAMEWORK_NAME}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${MACOS_MIN_OS_VERSION}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>DTPlatformName</key>
    <string>macosx</string>
    <key>DTSDKName</key>
    <string>macosx${MACOS_MIN_OS_VERSION}</string>
</dict>
</plist>
EOF

echo "==> Linking dynamic framework binary"
OUTPUT_LIB="${FW_VERSIONED}/${FRAMEWORK_NAME}"
xcrun -sdk macosx clang++ -dynamiclib \
    -isysroot "${MACOS_SDK}" \
    -arch arm64 -arch x86_64 \
    -mmacosx-version-min="${MACOS_MIN_OS_VERSION}" \
    -Wl,-force_load,"${TMP_DIR}/combined.a" \
    -framework Foundation -framework Metal -framework Accelerate -framework CoreML \
    -install_name "@rpath/${FRAMEWORK_NAME}.framework/Versions/Current/${FRAMEWORK_NAME}" \
    -o "${OUTPUT_LIB}"

echo "==> Generating dSYM"
DSYM_DIR="${BUILD_DIR}/dSYMs"
mkdir -p "${DSYM_DIR}"
xcrun dsymutil "${OUTPUT_LIB}" -o "${DSYM_DIR}/${FRAMEWORK_NAME}.dSYM"
# Strip in place so the shipped framework doesn't carry debug symbols (dSYM has them)
xcrun strip -S "${OUTPUT_LIB}" -o "${TMP_DIR}/stripped"
mv "${TMP_DIR}/stripped" "${OUTPUT_LIB}"

echo "==> Creating xcframework"
xcodebuild -create-xcframework \
    -framework "$(pwd)/${FW_ROOT}" \
    -debug-symbols "$(pwd)/${DSYM_DIR}/${FRAMEWORK_NAME}.dSYM" \
    -output "$(pwd)/${OUT_DIR}/${FRAMEWORK_NAME}.xcframework"

rm -rf "${TMP_DIR}"

echo
echo "Done: $(pwd)/${OUT_DIR}/${FRAMEWORK_NAME}.xcframework"
