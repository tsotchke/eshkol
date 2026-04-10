# typed: strict
# frozen_string_literal: true

# Homebrew formula for Eshkol
# This file should be copied to a homebrew-eshkol tap repository
#
# Usage:
#   brew tap tsotchke/eshkol
#   brew install eshkol
#
class Eshkol < Formula
  desc "Functional programming language with HoTT types and autodiff"
  homepage "https://eshkol.ai"
  url "https://github.com/tsotchke/eshkol/archive/refs/tags/v1.1.13-accelerate.tar.gz"
  # sha256 is filled in by scripts/update-homebrew-formula.sh after the release tarball is published
  sha256 "f2ad8b78d2bb4e5564365c3402b7275bcee9ce46d9aef1e7ffb626907662a531"
  license "MIT"
  head "https://github.com/tsotchke/eshkol.git", branch: "master"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  # Eshkol pins to LLVM 21 specifically: the codegen uses Triple::isOSWindows()
  # and Intrinsic::sponentry/frameaddress lowering that landed in LLVM 21,
  # and the bytecode VM ABI assumes LLVM 21's tagged_value_t struct layout.
  depends_on "llvm@21"
  depends_on "readline"

  def install
    # Set LLVM paths for build and runtime
    llvm = Formula["llvm@21"]

    # Refuse to build against anything older than 21.1.0 — earlier 21.0.x
    # snapshots are missing the AArch64 setjmp lowering Eshkol depends on.
    odie "Eshkol requires LLVM 21.1.0 or newer; found #{llvm.version}" if llvm.version < Version.new("21.1.0")

    ENV["PATH"] = "#{llvm.opt_bin}:#{ENV.fetch("PATH", nil)}"
    ENV["LDFLAGS"] = "-L#{llvm.opt_lib} -Wl,-rpath,#{llvm.opt_lib} #{ENV.fetch("LDFLAGS", nil)}"
    ENV["CPPFLAGS"] = "-I#{llvm.opt_include} #{ENV.fetch("CPPFLAGS", nil)}"

    # Set runtime library path so eshkol-run can find LLVM when generating stdlib.o
    ENV["DYLD_FALLBACK_LIBRARY_PATH"] = llvm.opt_lib

    # Configure with explicit LLVM paths and proper RPATH
    # CMAKE_BUILD_WITH_INSTALL_RPATH ensures rpath is set at build time (needed for stdlib.o generation)
    system "cmake", "-B", "build", "-G", "Ninja",
           "-DCMAKE_BUILD_TYPE=Release",
           "-DLLVM_DIR=#{llvm.opt_lib}/cmake/llvm",
           "-DCMAKE_INSTALL_RPATH=#{llvm.opt_lib}",
           "-DCMAKE_BUILD_RPATH=#{llvm.opt_lib}",
           "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
           "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
           "-DCMAKE_MACOSX_RPATH=ON",
           *std_cmake_args

    # Build eshkol-run first (without stdlib to avoid chicken-egg problem)
    system "cmake", "--build", "build", "--target", "eshkol-run"
    system "cmake", "--build", "build", "--target", "eshkol-repl"
    system "cmake", "--build", "build", "--target", "eshkol-static"

    # Compile stdlib using eshkol-run. The cmake build target above also produces
    # both build/stdlib.o (object code) and build/stdlib.bc (LLVM bitcode for the
    # REPL JIT's symbol discovery), so we don't actually need to invoke eshkol-run
    # again — but we keep this as a sanity check that the freshly-installed
    # eshkol-run binary works.
    system "cmake", "--build", "build", "--target", "stdlib"

    odie "stdlib.o was not created - compilation failed" unless File.exist?("build/stdlib.o")
    odie "stdlib.bc was not created - REPL JIT will lack symbol discovery" unless File.exist?("build/stdlib.bc")

    # Install binaries
    bin.install "build/eshkol-run"
    bin.install "build/eshkol-repl"

    # Install library files to lib/eshkol/ (primary location)
    (lib/"eshkol").mkpath
    (lib/"eshkol").install "build/stdlib.o"
    (lib/"eshkol").install "build/stdlib.bc"
    (lib/"eshkol").install "build/libeshkol-static.a"

    # Create symlinks in lib/ for convenience
    lib.install_symlink(lib/"eshkol/stdlib.o")
    lib.install_symlink(lib/"eshkol/stdlib.bc")
    lib.install_symlink(lib/"eshkol/libeshkol-static.a")

    # Install library source files
    (share/"eshkol").install "lib/stdlib.esk"
    (share/"eshkol/core").install Dir["lib/core/*"] if Dir.exist?("lib/core")
    (share/"eshkol/math").install Dir["lib/math/*"] if Dir.exist?("lib/math")
    (share/"eshkol").install "lib/math.esk" if File.exist?("lib/math.esk")
    (share/"eshkol/signal").install Dir["lib/signal/*"] if Dir.exist?("lib/signal")
    (share/"eshkol/ml").install Dir["lib/ml/*"] if Dir.exist?("lib/ml")
    (share/"eshkol/random").install Dir["lib/random/*"] if Dir.exist?("lib/random")
    (share/"eshkol/web").install Dir["lib/web/*"] if Dir.exist?("lib/web")
    (share/"eshkol/tensor").install Dir["lib/tensor/*"] if Dir.exist?("lib/tensor")
    (share/"eshkol/quantum").install Dir["lib/quantum/*"] if Dir.exist?("lib/quantum")
  end

  def caveats
    <<~EOS
      Eshkol has been installed!

      To start the interactive REPL:
        eshkol-repl

      To compile and run a program:
        eshkol-run yourfile.esk

      Documentation: https://eshkol.ai/
    EOS
  end

  test do
    # Test basic compilation
    (testpath/"hello.esk").write('(display "Hello, World!")')
    system "#{bin}/eshkol-run", "hello.esk", "-L#{lib}"
    assert_path_exists testpath/"a.out"
  end
end
