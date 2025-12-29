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
  url "https://github.com/tsotchke/eshkol/archive/v1.0.1.1.tar.gz"
  sha256 "PLACEHOLDER_UPDATE_AFTER_RELEASE"
  license "MIT"
  head "https://github.com/tsotchke/eshkol.git", branch: "master"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "llvm@17"
  depends_on "readline"

  def install
    # Set LLVM paths for build and runtime
    llvm = Formula["llvm@17"]
    ENV["PATH"] = "#{llvm.opt_bin}:#{ENV["PATH"]}"
    ENV["LDFLAGS"] = "-L#{llvm.opt_lib} -Wl,-rpath,#{llvm.opt_lib} #{ENV["LDFLAGS"]}"
    ENV["CPPFLAGS"] = "-I#{llvm.opt_include} #{ENV["CPPFLAGS"]}"

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
           "-DCMAKE_MACOSX_RPATH=ON",
           *std_cmake_args

    # Build eshkol-run, eshkol-repl, and static library (skip stdlib target - we'll generate it manually)
    system "cmake", "--build", "build", "--target", "eshkol-run"
    system "cmake", "--build", "build", "--target", "eshkol-repl"
    system "cmake", "--build", "build", "--target", "eshkol-static"

    # Generate stdlib.o using the freshly built eshkol-run compiler
    # This compiles lib/stdlib.esk to build/stdlib.o
    # Use --no-stdlib to avoid chicken-and-egg problem
    system "build/eshkol-run", "--no-stdlib", "--shared-lib", "-o", "build/stdlib", "lib/stdlib.esk"

    # Verify stdlib.o was created
    odie "stdlib.o was not created - eshkol-run compilation failed" unless File.exist?("build/stdlib.o")

    # Install binaries
    bin.install "build/eshkol-run"
    bin.install "build/eshkol-repl"

    # Install library files
    lib.install "build/stdlib.o"
    lib.install "build/libeshkol-static.a"
    (lib/"eshkol").install "build/stdlib.o"
    (lib/"eshkol").install "build/libeshkol-static.a"

    # Install library source files
    (share/"eshkol").install "lib/stdlib.esk"
    (share/"eshkol/core").install Dir["lib/core/*"] if Dir.exist?("lib/core")
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
    assert_predicate testpath/"a.out", :exist?
  end
end
