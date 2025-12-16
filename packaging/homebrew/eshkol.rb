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
  url "https://github.com/tsotchke/eshkol/archive/v1.0.1.tar.gz"
  sha256 "8d818fbd4c5c8033e3ec0006aeacc6e0824487aad6b14e79c57770c76ddd7975"
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

    # Build (stdlib.o is generated as part of this - eshkol-run compiles stdlib.esk)
    system "cmake", "--build", "build"

    # Verify stdlib.o was created
    odie "stdlib.o was not created - eshkol-run may have failed to find LLVM libraries" unless File.exist?("build/stdlib.o")

    # Install binaries
    bin.install "build/eshkol-run"
    bin.install "build/eshkol-repl"

    # Install library files
    lib.install "build/stdlib.o"
    (lib/"eshkol").install "build/stdlib.o"

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

      Documentation: https://eshkol.ai/docs
    EOS
  end

  test do
    # Test basic compilation
    (testpath/"hello.esk").write('(display "Hello, World!")')
    system "#{bin}/eshkol-run", "hello.esk", "-L#{lib}"
    assert_predicate testpath/"a.out", :exist?
  end
end
