# typed: false
# frozen_string_literal: true

# Homebrew formula for Eshkol
# This file is the canonical formula; it is mirrored to the tsotchke/homebrew-eshkol
# tap. The release workflow (.github/workflows/release.yml, bump-homebrew-tap job)
# rewrites ONLY the top-level `url` and top-level `sha256` on each release — the
# resource pins below are preserved.
#
# Usage:
#   brew tap tsotchke/eshkol
#   brew install eshkol
#
class Eshkol < Formula
  desc "Functional programming language with HoTT types and autodiff"
  homepage "https://eshkol.ai"
  url "https://github.com/tsotchke/eshkol/archive/refs/tags/v1.2.3-scale.tar.gz"
  # sha256 is filled in by scripts/update-homebrew-formula.sh after the release tarball is published
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/tsotchke/eshkol.git", branch: "master"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  # Eshkol pins to LLVM 21 specifically: the codegen uses Triple::isOSWindows()
  # and Intrinsic::sponentry/frameaddress lowering that landed in LLVM 21,
  # and the bytecode VM ABI assumes LLVM 21's tagged_value_t struct layout.
  depends_on "llvm@21"
  # PCRE2 powers the (require agent.regex) FFI. Homebrew forbids the vendored
  # FetchContent build during a formula build, so we link the real formula and
  # pass -DESHKOL_USE_SYSTEM_PCRE2=ON (see the install block).
  depends_on "pcre2"
  depends_on "readline"

  on_linux do
    # The native HTTP client links a minimal static libcurl on Linux (macOS uses
    # NSURLSession and needs no curl); curl's TLS backend here is OpenSSL, which
    # CMake's find_package(OpenSSL) locates via Homebrew's dependency search path.
    depends_on "openssl@3"

    resource "curl" do
      url "https://github.com/curl/curl.git", revision: "a05f34973e6c4bb629d018f7cb51487be1c904d8"
    end
  end

  # ──────────────────────────────────────────────────────────────────────────
  # Bundled agent-FFI native dependencies.
  #
  # Homebrew installs a CMake dependency provider that refuses any live
  # FetchContent population during a formula build. Instead of downloading these
  # at configure time, we provide each pinned dependency as a Homebrew `resource`
  # and stage it into FETCHCONTENT_SOURCE_DIR_<NAME> (see the install block). With
  # that override set, CMake uses the staged tree and never invokes the provider.
  #
  # The revisions below MUST match the GIT_TAG / URL pins in CMakeLists.txt.
  # ──────────────────────────────────────────────────────────────────────────
  resource "sqlite-amalgamation" do
    url "https://www.sqlite.org/2026/sqlite-amalgamation-3530300.zip"
    sha256 "646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431"
  end

  resource "zlib" do
    url "https://github.com/madler/zlib.git", revision: "51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf"
  end

  resource "tree-sitter" do
    url "https://github.com/tree-sitter/tree-sitter.git", revision: "cd5b087cd9f45ca6d93ab1954f6b7c8534f324d2"
  end

  resource "yoga" do
    url "https://github.com/facebook/yoga.git", revision: "042f5013152eb81c1552dec945b88f7b95ca350f"
  end

  # tree-sitter grammars: parser.c is checked in and compiled source-only.
  resource "ts-javascript" do
    url "https://github.com/tree-sitter/tree-sitter-javascript.git", revision: "58404d8cf191d69f2674a8fd507bd5776f46cb11"
  end
  resource "ts-typescript" do
    url "https://github.com/tree-sitter/tree-sitter-typescript.git", revision: "75b3874edb2dc714fb1fd77a32013d0f8699989f"
  end
  resource "ts-python" do
    url "https://github.com/tree-sitter/tree-sitter-python.git", revision: "26855eabccb19c6abf499fbc5b8dc7cc9ab8bc64"
  end
  resource "ts-rust" do
    url "https://github.com/tree-sitter/tree-sitter-rust.git", revision: "77a3747266f4d621d0757825e6b11edcbf991ca5"
  end
  resource "ts-go" do
    url "https://github.com/tree-sitter/tree-sitter-go.git", revision: "2346a3ab1bb3857b48b29d779a1ef9799a248cd7"
  end
  resource "ts-c" do
    url "https://github.com/tree-sitter/tree-sitter-c.git", revision: "b780e47fc780ddc8da13afa35a3f4ed5c157823d"
  end
  resource "ts-cpp" do
    url "https://github.com/tree-sitter/tree-sitter-cpp.git", revision: "8b5b49eb196bec7040441bee33b2c9a4838d6967"
  end
  resource "ts-java" do
    url "https://github.com/tree-sitter/tree-sitter-java.git", revision: "e10607b45ff745f5f876bfa3e94fbcc6b44bdc11"
  end
  resource "ts-ruby" do
    url "https://github.com/tree-sitter/tree-sitter-ruby.git", revision: "ad907a69da0c8a4f7a943a7fe012712208da6dee"
  end
  resource "ts-bash" do
    url "https://github.com/tree-sitter/tree-sitter-bash.git", revision: "a06c2e4415e9bc0346c6b86d401879ffb44058f7"
  end

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

    # Stage the bundled agent-FFI dependency sources so CMake resolves them from
    # disk instead of populating them with FetchContent (which Homebrew forbids).
    # The FETCHCONTENT_SOURCE_DIR_<NAME> keys are the uppercased FetchContent
    # content names declared in CMakeLists.txt.
    deps_root = buildpath/"agent-ffi-deps"
    staged = {
      "ESHKOL_SQLITE_AMALGAMATION" => "sqlite-amalgamation",
      "ESHKOL_ZLIB"                => "zlib",
      "ESHKOL_TREE_SITTER"         => "tree-sitter",
      "ESHKOL_YOGA"                => "yoga",
      "ESHKOL_TS_JAVASCRIPT"       => "ts-javascript",
      "ESHKOL_TS_TYPESCRIPT"       => "ts-typescript",
      "ESHKOL_TS_PYTHON"           => "ts-python",
      "ESHKOL_TS_RUST"             => "ts-rust",
      "ESHKOL_TS_GO"               => "ts-go",
      "ESHKOL_TS_C"                => "ts-c",
      "ESHKOL_TS_CPP"              => "ts-cpp",
      "ESHKOL_TS_JAVA"             => "ts-java",
      "ESHKOL_TS_RUBY"             => "ts-ruby",
      "ESHKOL_TS_BASH"             => "ts-bash",
    }
    staged["ESHKOL_CURL"] = "curl" if OS.linux?

    fetchcontent_args = staged.map do |var, res_name|
      target = deps_root/res_name
      resource(res_name).stage(target.to_s)
      # SQLite is consumed as ${SOURCE_DIR}/sqlite3.c; point at the directory
      # that actually contains it (the zip may unpack into a subdirectory).
      src_dir = target
      if var == "ESHKOL_SQLITE_AMALGAMATION"
        sqlite_c = Dir[target/"**/sqlite3.c"].first
        odie "staged sqlite amalgamation is missing sqlite3.c" if sqlite_c.nil?
        src_dir = Pathname.new(sqlite_c).dirname
      end
      "-DFETCHCONTENT_SOURCE_DIR_#{var}=#{src_dir}"
    end

    homebrew_dep_args = [
      "-DESHKOL_HOMEBREW_BUILD=ON",
      "-DESHKOL_USE_SYSTEM_PCRE2=ON",
      "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    ]

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
           *homebrew_dep_args,
           *fetchcontent_args,
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

    # Install library files to lib/eshkol/ (primary location).
    # eshkol-run resolves its runtime archive as <prefix>/lib/eshkol; it prefers
    # libeshkol-runtime.a and falls back to the legacy libeshkol-static.a, so the
    # PRIMARY runtime archive (libeshkol-runtime.a) must be installed too —
    # without it, stdlib.o's runtime references are unresolved and no user
    # program links.
    (lib/"eshkol").mkpath
    (lib/"eshkol").install "build/stdlib.o"
    (lib/"eshkol").install "build/stdlib.bc"
    (lib/"eshkol").install "build/libeshkol-runtime.a"
    (lib/"eshkol").install "build/libeshkol-static.a"

    # eshkol-run force-loads libeshkol-agent-ffi.a and replays its dependency
    # closure (the stable-named eshkol-agent-*.a archives) at AOT link time for
    # programs that (require agent.*). eshkol-run resolves all of them beside the
    # runtime archive, so they must land in the same lib/eshkol directory.
    (lib/"eshkol").install "build/libeshkol-agent-ffi.a" if File.exist?("build/libeshkol-agent-ffi.a")
    Dir["build/eshkol-agent-*.a"].each { |archive| (lib/"eshkol").install archive }

    # Create symlinks in lib/ for convenience
    lib.install_symlink(lib/"eshkol/stdlib.o")
    lib.install_symlink(lib/"eshkol/stdlib.bc")
    lib.install_symlink(lib/"eshkol/libeshkol-runtime.a")
    lib.install_symlink(lib/"eshkol/libeshkol-static.a")

    # Install module sources. eshkol-run resolves (require …) against the
    # directory that contains stdlib.esk, discovered as <prefix>/share/eshkol/lib.
    # A dotted module maps under it (agent.regex -> share/eshkol/lib/agent/regex.esk),
    # so every runtime module must live below share/eshkol/lib — not share/eshkol.
    esklib = share/"eshkol/lib"
    esklib.mkpath
    esklib.install "lib/stdlib.esk"
    esklib.install "lib/math.esk" if File.exist?("lib/math.esk")
    esklib.install "lib/tensorcore.esk" if File.exist?("lib/tensorcore.esk")
    # core carries nested submodules (core/…/….esk); install the whole tree.
    %w[core math signal ml random web tensor quantum].each do |mod|
      esklib.install "lib/#{mod}" if Dir.exist?("lib/#{mod}")
    end
    # Agent modules: install the .esk wrappers only (skip the bundled C sources
    # under lib/agent/c) so (require agent.regex) etc. resolve.
    (esklib/"agent").install Dir["lib/agent/*.esk"] if Dir.exist?("lib/agent")
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
    # Basic compilation + execution.
    (testpath/"hello.esk").write('(display "Hello, World!")(newline)')
    system bin/"eshkol-run", "hello.esk"
    assert_path_exists testpath/"a.out"
    assert_equal "Hello, World!", shell_output("#{testpath}/a.out").strip

    # (require …) must resolve against the installed module tree, and the runtime
    # archive must link — a keg missing libeshkol-runtime.a or the share/eshkol/lib
    # module sources fails here.
    (testpath/"mods.esk").write("(require stdlib)(display (length (list 1 2 3)))(newline)")
    system bin/"eshkol-run", "mods.esk", "-o", "mods"
    assert_equal "3", shell_output("#{testpath}/mods").strip
  end
end
