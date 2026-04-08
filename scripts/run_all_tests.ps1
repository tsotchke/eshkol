param(
    [string]$BuildDir = "",
    [switch]$SkipConfigureBuild,
    [ValidateSet("all", "xla", "gpu")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""
}

function New-SuiteState {
    param([string]$Name)
    [pscustomobject]@{
        Name      = $Name
        Passed    = 0
        Failed    = 0
        Skipped   = 0
        Failures  = New-Object System.Collections.Generic.List[string]
        FailureLog = New-Object System.Collections.Generic.List[string]
    }
}

function Add-Pass {
    param($Suite)
    $Suite.Passed++
}

function Add-Fail {
    param(
        $Suite,
        [string]$Failure
    )
    $Suite.Failed++
    if ($Failure) {
        $Suite.Failures.Add($Failure) | Out-Null
    }
}

function Add-Skip {
    param(
        $Suite,
        [string]$Reason
    )
    $Suite.Skipped++
    if ($Reason) {
        $Suite.FailureLog.Add($Reason) | Out-Null
    }
}

function Show-SuiteSummary {
    param(
        $Suite,
        [string]$Title = "Test Results Summary"
    )

    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    $total = $Suite.Passed + $Suite.Failed + $Suite.Skipped
    Write-Host ("Total Tests:    {0}" -f $total)
    Write-Host ("Passed:         {0}" -f $Suite.Passed) -ForegroundColor Green
    Write-Host ("Failed:         {0}" -f $Suite.Failed) -ForegroundColor Red
    if ($Suite.Skipped -gt 0) {
        Write-Host ("Skipped:        {0}" -f $Suite.Skipped) -ForegroundColor Yellow
    }
    if ($Suite.Failures.Count -gt 0) {
        Write-Host ""
        Write-Host "Failed Tests:"
        foreach ($failure in $Suite.Failures) {
            Write-Host ("  - {0}" -f $failure)
        }
    }
    if ($Suite.FailureLog.Count -gt 0) {
        Write-Host ""
        foreach ($line in $Suite.FailureLog) {
            Write-Host $line
        }
    }
    Write-Host ""
}

function Get-ProjectRoot {
    Split-Path -Parent $PSScriptRoot
}

function Resolve-BuildDirectory {
    param(
        [string]$ProjectRoot,
        [string]$RequestedBuildDir
    )

    $candidates = @()
    if ($RequestedBuildDir) {
        $candidates += $RequestedBuildDir
    }
    $candidates += @(
        (Join-Path $ProjectRoot "build"),
        "C:\src\eshkol-win-suite",
        "C:\src\eshkol-win-smoke-ninja"
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }

        $resolvedCandidate = $null
        try {
            $resolvedCandidate = (Resolve-Path $candidate -ErrorAction Stop).Path
        } catch {
            continue
        }

        foreach ($binDir in @(
            $resolvedCandidate,
            (Join-Path $resolvedCandidate "Release"),
            (Join-Path $resolvedCandidate "Debug")
        )) {
            if (Test-Path (Join-Path $binDir "eshkol-run.exe")) {
                return [pscustomobject]@{
                    RootDir   = $resolvedCandidate
                    BinaryDir = $binDir
                }
            }
        }
    }

    throw "No native Windows build directory with eshkol-run.exe was found. Pass -BuildDir or build the native tree first."
}

function Ensure-NativeTempRoot {
    param([string]$TempRoot)
    New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
    New-Item -ItemType Directory -Force -Path "C:\tmp" | Out-Null
    New-Item -ItemType Directory -Force -Path "C:\tmp\eshkol_examples_test" | Out-Null
}

function New-OutputBase {
    param(
        [string]$TempRoot,
        [string]$SuiteName,
        [string]$TestName
    )

    $safeSuite = ($SuiteName -replace "[^A-Za-z0-9_.-]", "_")
    $safeName = ($TestName -replace "[^A-Za-z0-9_.-]", "_")
    $dir = Join-Path $TempRoot $safeSuite
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Join-Path $dir $safeName
}

function Get-Text {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return ""
    }
    return [System.IO.File]::ReadAllText($Path)
}

function Invoke-ProcessCapture {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory,
        [int]$TimeoutSec = 0,
        [string]$InputText = ""
    )

    $effectiveWorkingDirectory = $WorkingDirectory
    if ($effectiveWorkingDirectory -like "\\\\*") {
        $effectiveWorkingDirectory = $script:BuildDir
    }

    if ($TimeoutSec -le 0) {
        Push-Location $effectiveWorkingDirectory
        try {
            $previousErrorAction = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                if ($InputText -ne "") {
                    $outputLines = $InputText | & $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() }
                } else {
                    $outputLines = & $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() }
                }
                $output = $outputLines -join [Environment]::NewLine
                $exitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorAction
            }
        } finally {
            Pop-Location
        }
        [pscustomobject]@{
            ExitCode = $exitCode
            TimedOut = $false
            StdOut   = $output
            StdErr   = ""
            Output   = $output
        }
        return
    }

    $job = Start-Job -ScriptBlock {
        param($FilePath, $Arguments, $WorkingDirectory, $InputText)
        Set-Location $WorkingDirectory
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            if ($InputText -ne "") {
                $outputLines = $InputText | & $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() }
            } else {
                $outputLines = & $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() }
            }
            [pscustomobject]@{
                ExitCode = $LASTEXITCODE
                Output   = $outputLines -join [Environment]::NewLine
            }
        } finally {
            $ErrorActionPreference = $previousErrorAction
        }
    } -ArgumentList @($FilePath, $Arguments, $effectiveWorkingDirectory, $InputText)

    try {
        $completed = Wait-Job -Id $job.Id -Timeout $TimeoutSec
        if (-not $completed) {
            Stop-Job -Id $job.Id -ErrorAction SilentlyContinue
            Receive-Job -Id $job.Id -ErrorAction SilentlyContinue | Out-Null
            [pscustomobject]@{
                ExitCode = 124
                TimedOut = $true
                StdOut   = ""
                StdErr   = ""
                Output   = ""
            }
            return
        }

        $result = Receive-Job -Id $job.Id
        [pscustomobject]@{
            ExitCode = if ($null -ne $result.ExitCode) { $result.ExitCode } else { 1 }
            TimedOut = $false
            StdOut   = $result.Output
            StdErr   = ""
            Output   = $result.Output
        }
    } finally {
        Remove-Job -Id $job.Id -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-EshkolCompile {
    param(
        [string]$EshkolRun,
        [string]$ProjectRoot,
        [string]$BuildDir,
        [string]$TestFile,
        [string]$OutputBase,
        [string[]]$ExtraArgs = @()
    )

    $exePath = $OutputBase + ".exe"
    Remove-Item $exePath, $OutputBase -Force -ErrorAction SilentlyContinue

    $args = @($TestFile, "-L", $BuildDir)
    if ($ExtraArgs.Count -gt 0) {
        $args += $ExtraArgs
    }
    $args += @("-o", $OutputBase)

    Push-Location $BuildDir
    try {
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $outputLines = & $EshkolRun @args 2>&1 | ForEach-Object { $_.ToString() }
            $exitCode = $LASTEXITCODE
            $outputText = ($outputLines -join [Environment]::NewLine)
        } finally {
            $ErrorActionPreference = $previousErrorAction
        }
    } finally {
        Pop-Location
    }

    [pscustomobject]@{
        ExitCode = $exitCode
        Success  = ($exitCode -eq 0 -and (Test-Path $exePath))
        Output   = $outputText
        ExePath  = $exePath
    }
}

function Get-TestFiles {
    param(
        [string]$ProjectRoot,
        [string[]]$Patterns,
        [switch]$Recurse
    )

    $files = New-Object System.Collections.Generic.List[string]
    foreach ($pattern in $Patterns) {
        $glob = Join-Path $ProjectRoot $pattern
        if ($Recurse) {
            $items = Get-ChildItem -Path $glob -File -Recurse -ErrorAction SilentlyContinue
        } else {
            $items = Get-ChildItem -Path $glob -File -ErrorAction SilentlyContinue
        }
        foreach ($item in $items) {
            $files.Add($item.FullName) | Out-Null
        }
    }
    return ,@($files | Sort-Object -Unique)
}

function Format-TestStatus {
    param(
        [string]$TestName,
        [string]$Status,
        [ConsoleColor]$Color
    )
    Write-Host ("Testing {0,-50} {1}" -f $TestName, $Status) -ForegroundColor $Color
}

function Invoke-SimpleCompileRunSuite {
    param(
        [string]$SuiteName,
        [string]$Title,
        [string[]]$Patterns,
        [string]$FailRegex = "",
        [string]$RuntimeErrorRegex = "",
        [int]$TimeoutSec = 0,
        [switch]$Recurse
    )

    Write-Section $Title
    $suite = New-SuiteState $SuiteName
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns $Patterns -Recurse:$Recurse

    if ($files.Count -eq 0) {
        Add-Skip $suite "No test files found."
        Show-SuiteSummary $suite
        return $suite
    }

    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName $SuiteName -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase
        if (-not $compile.Success) {
            Format-TestStatus $testName "COMPILE FAIL" Red
            Add-Fail $suite $testName
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot -TimeoutSec $TimeoutSec
        if ($run.TimedOut) {
            Format-TestStatus $testName "TIMEOUT" Yellow
            Add-Fail $suite "$testName (timeout)"
            continue
        }
        if ($run.ExitCode -ne 0) {
            Format-TestStatus $testName ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) Red
            Add-Fail $suite $testName
            continue
        }
        if ($FailRegex -and $run.Output -match $FailRegex) {
            Format-TestStatus $testName "ASSERTION FAIL" Red
            Add-Fail $suite $testName
            continue
        }
        if ($RuntimeErrorRegex -and $run.Output -match $RuntimeErrorRegex) {
            Format-TestStatus $testName "RUNTIME ERROR" Yellow
            Add-Fail $suite $testName
            continue
        }

        Format-TestStatus $testName "PASS" Green
        Add-Pass $suite
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-MemorySuite {
    Write-Section "Eshkol Memory Test Suite"
    $suite = New-SuiteState "memory"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/memory/*.esk")

    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $source = Get-Text $testFile
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "memory" -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase

        if ($source -match ";;; Expected: Error") {
            if ($compile.Success) {
                Format-TestStatus $testName "SHOULD HAVE FAILED" Red
                Add-Fail $suite "$testName (expected compile error)"
            } else {
                Format-TestStatus $testName "PASS (expected error)" Green
                Add-Pass $suite
            }
            continue
        }

        if (-not $compile.Success) {
            Format-TestStatus $testName "COMPILE FAIL" Red
            Add-Fail $suite $testName
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -ne 0) {
            Format-TestStatus $testName ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) Red
            Add-Fail $suite $testName
        } elseif ($run.Output -match "error:") {
            Format-TestStatus $testName "RUNTIME ERROR" Yellow
            Add-Fail $suite $testName
        } else {
            Format-TestStatus $testName "PASS" Green
            Add-Pass $suite
        }
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-ModulesSuite {
    Write-Section "Eshkol Modules Test Suite"
    $suite = New-SuiteState "modules"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/modules/*.esk")

    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $source = Get-Text $testFile
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "modules" -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase

        if ($source -match ";;; Expected: Error") {
            $hasCompileError = (-not $compile.Success) -or ($compile.Output -match "(?i)error:")
            if ($hasCompileError -and (-not (Test-Path $compile.ExePath))) {
                Format-TestStatus $testName "PASS (expected compile error)" Green
                Add-Pass $suite
                continue
            }
            if ($hasCompileError) {
                $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot
                if ($run.ExitCode -ne 0) {
                    Format-TestStatus $testName "PASS (expected runtime error)" Green
                    Add-Pass $suite
                } else {
                    Format-TestStatus $testName "SHOULD HAVE FAILED" Red
                    Add-Fail $suite "$testName (expected error)"
                }
                continue
            }
            Format-TestStatus $testName "SHOULD HAVE FAILED" Red
            Add-Fail $suite "$testName (expected error)"
            continue
        }

        if (-not $compile.Success) {
            Format-TestStatus $testName "COMPILE FAIL" Red
            Add-Fail $suite $testName
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -ne 0) {
            Format-TestStatus $testName ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) Red
            Add-Fail $suite $testName
        } elseif ($run.Output -match "error:") {
            Format-TestStatus $testName "RUNTIME ERROR" Yellow
            Add-Fail $suite $testName
        } else {
            Format-TestStatus $testName "PASS" Green
            Add-Pass $suite
        }
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-ParserSuite {
    Write-Section "Eshkol Parser Test Suite"
    $suite = New-SuiteState "parser"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/parser/*.esk")

    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $source = Get-Text $testFile
        $expectsError = ($source -match ";;; Expected: Error")
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "parser" -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase

        if (-not $compile.Success) {
            if ($expectsError) {
                Format-TestStatus $testName "PASS (expected error)" Green
                Add-Pass $suite
            } else {
                Format-TestStatus $testName "COMPILE FAIL" Red
                Add-Fail $suite $testName
            }
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -ne 0) {
            if ($expectsError) {
                Format-TestStatus $testName "PASS (expected error)" Green
                Add-Pass $suite
            } else {
                Format-TestStatus $testName ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) Red
                Add-Fail $suite $testName
            }
        } elseif ($run.Output -match "error:") {
            Format-TestStatus $testName "RUNTIME ERROR" Yellow
            Add-Fail $suite $testName
        } else {
            Format-TestStatus $testName "PASS" Green
            Add-Pass $suite
        }
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-TypesystemSuite {
    Write-Section "Eshkol Type System Test Suite"
    $suite = New-SuiteState "typesystem"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/typesystem/*.esk")

    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $sourceLines = Get-Content $testFile
        $modeLine = $sourceLines | Where-Object { $_ -match '^;; EXPECT-MODE:' } | Select-Object -First 1
        $mode = ""
        if ($modeLine) {
            $mode = ($modeLine -replace '^;; EXPECT-MODE:\s*', '')
        }

        $extraArgs = @()
        switch ($mode) {
            "strict-types" { $extraArgs += "--strict-types" }
            "unsafe"       { $extraArgs += "--unsafe" }
        }

        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "typesystem" -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase -ExtraArgs $extraArgs
        $stderr = $compile.Output
        $passed = $true

        foreach ($line in $sourceLines) {
            if ($line -match '^;; EXPECT-STDERR:') {
                $pattern = ($line -replace '^;; EXPECT-STDERR:\s*', '')
                if ($pattern -and $stderr.IndexOf($pattern, [System.StringComparison]::Ordinal) -lt 0) {
                    $passed = $false
                }
            }
            if ($line -match '^;; EXPECT-NO-STDERR:') {
                $pattern = ($line -replace '^;; EXPECT-NO-STDERR:\s*', '')
                if ($pattern -and $stderr.IndexOf($pattern, [System.StringComparison]::Ordinal) -ge 0) {
                    $passed = $false
                }
            }
        }

        if ($passed) {
            Format-TestStatus $testName "PASS" Green
            Add-Pass $suite
        } else {
            Format-TestStatus $testName "FAIL" Red
            Add-Fail $suite $testName
        }
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-ReplSuite {
    Write-Section "Eshkol REPL Test Suite"
    $suite = New-SuiteState "repl"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/repl/*.esk")

    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $inputText = (Get-Text $testFile) + [Environment]::NewLine + "exit" + [Environment]::NewLine
        $run = Invoke-ProcessCapture -FilePath $script:EshkolRepl -WorkingDirectory $script:ProjectRoot -TimeoutSec 10 -InputText $inputText

        if ($run.TimedOut) {
            Format-TestStatus $testName "TIMEOUT" Yellow
            Add-Fail $suite "$testName (timeout)"
        } elseif ($run.Output -match "error:") {
            Format-TestStatus $testName "RUNTIME ERROR" Red
            Add-Fail $suite $testName
        } elseif ($run.Output -match "Segmentation fault") {
            Format-TestStatus $testName "SEGFAULT" Red
            Add-Fail $suite $testName
        } else {
            Format-TestStatus $testName "PASS" Green
            Add-Pass $suite
        }
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-LogicSuite {
    Write-Section "Eshkol Consciousness Engine Tests"
    $suite = New-SuiteState "logic"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/logic/*.esk")

    foreach ($testFile in $files) {
        $testName = [System.IO.Path]::GetFileNameWithoutExtension($testFile)
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "logic" -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase

        if (-not $compile.Success) {
            Write-Host ("  {0,-30} FAIL (compile error)" -f $testName) -ForegroundColor Red
            Add-Fail $suite "$testName (compile)"
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -eq 0) {
            Write-Host ("  {0,-30} PASS" -f $testName) -ForegroundColor Green
            Add-Pass $suite
        } else {
            Write-Host ("  {0,-30} FAIL (runtime error)" -f $testName) -ForegroundColor Red
            Add-Fail $suite "$testName (runtime)"
        }
    }

    Show-SuiteSummary $suite "Results Summary"
    return $suite
}

function Invoke-TimedCompileRunSuite {
    param(
        [string]$SuiteName,
        [string]$Title,
        [string[]]$Patterns,
        [int]$TimeoutSec,
        [string]$FailRegex,
        [switch]$RequirePassMarker
    )

    Write-Section $Title
    $suite = New-SuiteState $SuiteName
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns $Patterns

    foreach ($testFile in $files) {
        $testName = [System.IO.Path]::GetFileNameWithoutExtension($testFile)
        Write-Host -NoNewline ("  {0} ... " -f $testName)
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName $SuiteName -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase

        if (-not $compile.Success) {
            Write-Host "COMPILE FAIL" -ForegroundColor Red
            Add-Fail $suite "$testName (compile)"
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot -TimeoutSec $TimeoutSec
        if ($run.TimedOut) {
            Write-Host ("TIMEOUT ({0}s)" -f $TimeoutSec) -ForegroundColor Red
            Add-Fail $suite "$testName (timeout)"
            continue
        }
        if ($run.ExitCode -ne 0) {
            Write-Host ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) -ForegroundColor Red
            Add-Fail $suite "$testName (runtime exit $($run.ExitCode))"
            continue
        }
        if ($FailRegex -and $run.Output -match $FailRegex) {
            Write-Host "FAIL" -ForegroundColor Red
            Add-Fail $suite "$testName (wrong result)"
            continue
        }
        if ($RequirePassMarker -and $run.Output -notmatch "PASS") {
            Write-Host "UNKNOWN" -ForegroundColor Yellow
            Add-Fail $suite "$testName (no PASS/FAIL)"
            continue
        }
        Write-Host "PASS" -ForegroundColor Green
        Add-Pass $suite
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-ExamplesSuite {
    Write-Section "Eshkol Examples Test Suite"
    $suite = New-SuiteState "examples"
    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("examples/*.esk")

    $current = 0
    $total = $files.Count
    foreach ($testFile in $files) {
        $current++
        $testName = Split-Path -Leaf $testFile
        if ($testName -like "selene_*" -or
            $testName -like "qllm_*" -or
            $testName -eq "agent.esk" -or
            $testName -like "consciousness_*") {
            Write-Host ("[{0,3}/{1,3}] {2,-50} SKIP (proprietary)" -f $current, $total, $testName) -ForegroundColor Yellow
            Add-Skip $suite "$testName (proprietary)"
            continue
        }

        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "examples" -TestName $testName
        Write-Host -NoNewline ("[{0,3}/{1,3}] {2,-50} " -f $current, $total, $testName)
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase

        if (-not $compile.Success) {
            Write-Host "COMPILE FAIL" -ForegroundColor Red
            Add-Fail $suite $testName
            continue
        }

        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -ne 0) {
            Write-Host ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) -ForegroundColor Red
            Add-Fail $suite "$testName (exit $($run.ExitCode))"
        } elseif ($run.Output -match "(?i)error|segmentation fault|abort") {
            Write-Host "RUNTIME ERROR" -ForegroundColor Yellow
            Add-Fail $suite $testName
        } else {
            Write-Host "PASS" -ForegroundColor Green
            Add-Pass $suite
        }
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-CppTypeSuite {
    Write-Section "Eshkol HoTT Type Checker C++ Tests"
    $suite = New-SuiteState "cpp_type"
    $llvmConfig = $null
    $requiredLlvmMajor = if ($env:ESHKOL_REQUIRED_LLVM_MAJOR) { $env:ESHKOL_REQUIRED_LLVM_MAJOR } else { "21" }

    $llvmCandidates = @((Join-Path $script:BuildDir "llvm-config.exe"))
    $llvmCandidates += Get-ChildItem -Path "C:\src" -Directory -Filter "clang+llvm-$requiredLlvmMajor*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "bin\llvm-config.exe" }
    $llvmCandidates += Get-ChildItem -Path "C:\src" -Directory -Filter "llvm-$requiredLlvmMajor*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "bin\llvm-config.exe" }
    foreach ($candidate in $llvmCandidates) {
        if ($candidate -and (Test-Path $candidate)) {
            $llvmConfig = $candidate
            break
        }
    }
    if (-not $llvmConfig) {
        $llvmConfig = (Get-Command "llvm-config.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
    }
    if (-not $llvmConfig) {
        Add-Skip $suite "llvm-config.exe not found; skipping C++ type tests."
        Show-SuiteSummary $suite "C++ Test Results Summary"
        return $suite
    }

    $llvmBinDir = Split-Path -Parent $llvmConfig
    $clangxx = Join-Path $llvmBinDir "clang++.exe"
    if (-not (Test-Path $clangxx)) {
        Add-Skip $suite "clang++.exe not found next to llvm-config.exe; skipping C++ type tests."
        Show-SuiteSummary $suite "C++ Test Results Summary"
        return $suite
    }

    $llvmCxxFlags = @()
    $llvmLdFlags = @()
    $llvmLibs = @()
    $llvmSystemLibs = @()

    $onWindows = ($env:OS -eq "Windows_NT")
    if (-not $onWindows) {
        $llvmCxxFlagsRaw = (& $llvmConfig --cxxflags) -replace "(^|\s)-std=[^\s]+", "" -replace "(^|\s)-fno-exceptions", ""
        $llvmLdFlagsRaw = & $llvmConfig --ldflags
        $llvmLibsRaw = & $llvmConfig --libs all
        $llvmSystemLibsRaw = & $llvmConfig --system-libs

        if ($llvmCxxFlagsRaw) { $llvmCxxFlags = [regex]::Split($llvmCxxFlagsRaw.Trim(), "\s+") | Where-Object { $_ } }
        if ($llvmLdFlagsRaw) { $llvmLdFlags = [regex]::Split($llvmLdFlagsRaw.Trim(), "\s+") | Where-Object { $_ } }
        if ($llvmLibsRaw) { $llvmLibs = [regex]::Split($llvmLibsRaw.Trim(), "\s+") | Where-Object { $_ } }
        if ($llvmSystemLibsRaw) { $llvmSystemLibs = [regex]::Split($llvmSystemLibsRaw.Trim(), "\s+") | Where-Object { $_ } }
    }

    $sources = @(
        (Join-Path $script:ProjectRoot "lib/types/hott_types.cpp"),
        (Join-Path $script:ProjectRoot "lib/types/type_checker.cpp"),
        (Join-Path $script:ProjectRoot "lib/types/dependent.cpp"),
        (Join-Path $script:ProjectRoot "lib/core/ast.cpp")
    )

    $tests = @(
        (Join-Path $script:ProjectRoot "tests/types/hott_types_test.cpp"),
        (Join-Path $script:ProjectRoot "tests/types/type_checker_test.cpp")
    )

    foreach ($testFile in $tests) {
        if (-not (Test-Path $testFile)) {
            continue
        }

        $testName = [System.IO.Path]::GetFileNameWithoutExtension($testFile)
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "cpp_type" -TestName $testName
        $exePath = $outputBase + ".exe"
        Remove-Item $exePath -Force -ErrorAction SilentlyContinue

        Write-Host ("Compiling {0,-40} " -f ($testName + "...")) -NoNewline
        $args = @(
            "-std=c++20",
            "-I" + (Join-Path $script:ProjectRoot "inc")
        ) + $llvmCxxFlags + $sources + @($testFile) + $llvmLdFlags + $llvmLibs + $llvmSystemLibs + @("-o", $exePath)
        Push-Location $script:BuildDir
        try {
            $previousErrorAction = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $compileOutput = (& $clangxx @args 2>&1 | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
                $compileExit = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorAction
            }
        } finally {
            Pop-Location
        }
        if ($compileExit -ne 0) {
            Write-Host "COMPILE FAILED" -ForegroundColor Red
            Add-Fail $suite "$testName (compile)"
            continue
        }
        Write-Host "OK" -ForegroundColor Green

        Write-Host ("Running   {0,-40} " -f ($testName + "...")) -NoNewline
        $run = Invoke-ProcessCapture -FilePath $exePath -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -eq 0) {
            Write-Host "PASSED" -ForegroundColor Green
            Add-Pass $suite
        } else {
            Write-Host "FAILED" -ForegroundColor Red
            Add-Fail $suite "$testName (runtime)"
        }
    }

    Show-SuiteSummary $suite "C++ Test Results Summary"
    return $suite
}

function Invoke-WebSuite {
    Write-Section "Eshkol Web/WASM Test Suite"
    $suite = New-SuiteState "web"

    if (-not (Test-Path $script:EshkolServer)) {
        Write-Host "Building eshkol-server..." -ForegroundColor Yellow
        & cmake --build $script:BuildDir --target eshkol-server --parallel
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $script:EshkolServer)) {
            Add-Fail $suite "eshkol-server build"
            Show-SuiteSummary $suite
            return $suite
        }
    }

    $webTests = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/web/*.esk")
    foreach ($testFile in $webTests) {
        $testName = Split-Path -Leaf $testFile
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "web" -TestName $testName
        $wasmPath = $outputBase + ".wasm"
        $args = @($testFile, "--wasm", "-o", $wasmPath)
        Push-Location $script:BuildDir
        try {
            $previousErrorAction = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                & $script:EshkolRun @args *> $null
                $exitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorAction
            }
        } finally {
            Pop-Location
        }

        if ($exitCode -eq 0 -and (Test-Path $wasmPath)) {
            $bytes = [System.IO.File]::ReadAllBytes($wasmPath)
            if ($bytes.Length -ge 4 -and $bytes[0] -eq 0x00 -and $bytes[1] -eq 0x61 -and $bytes[2] -eq 0x73 -and $bytes[3] -eq 0x6D) {
                Format-TestStatus $testName "PASS" Green
                Add-Pass $suite
            } else {
                Format-TestStatus $testName "WASM INVALID" Red
                Add-Fail $suite "$testName (invalid wasm)"
            }
        } else {
            Format-TestStatus $testName "COMPILE FAIL" Red
            Add-Fail $suite "$testName (wasm compile)"
        }
    }

    $serverStdout = Join-Path $script:TempRoot "eshkol-server.stdout.log"
    $serverStderr = Join-Path $script:TempRoot "eshkol-server.stderr.log"
    Remove-Item $serverStdout, $serverStderr -Force -ErrorAction SilentlyContinue
    $port = 19876
    $server = Start-Process -FilePath $script:EshkolServer -ArgumentList @("--port", "$port") -WorkingDirectory $script:ProjectRoot -RedirectStandardOutput $serverStdout -RedirectStandardError $serverStderr -PassThru
    Start-Sleep -Seconds 2
    $server.Refresh()
    if ($server.HasExited) {
        Add-Fail $suite "server start"
        Show-SuiteSummary $suite
        return $suite
    }

    try {
        $health = Invoke-RestMethod -Uri ("http://localhost:{0}/health" -f $port)
        if ($health.status -eq "ok") { Add-Pass $suite } else { Add-Fail $suite "health endpoint" }

        $compileResp = Invoke-RestMethod -Method Post -Uri ("http://localhost:{0}/compile" -f $port) -ContentType "application/json" -Body '{"code":"(define (square x) (* x x))","session_id":"test1"}'
        if ($compileResp.success -eq $true) { Add-Pass $suite } else { Add-Fail $suite "compile simple function" }

        $externResp = Invoke-RestMethod -Method Post -Uri ("http://localhost:{0}/compile" -f $port) -ContentType "application/json" -Body '{"code":"(extern i32 web-get-body :real web_get_body)\n(define (test) (web-get-body))","session_id":"test2"}'
        if ($externResp.success -eq $true) { Add-Pass $suite } else { Add-Fail $suite "compile with web externals" }

        $mathResp = Invoke-RestMethod -Method Post -Uri ("http://localhost:{0}/compile" -f $port) -ContentType "application/json" -Body '{"code":"(define (circle-area r) (* 3.14159 (* r r)))","session_id":"test3"}'
        if ($mathResp.success -eq $true) { Add-Pass $suite } else { Add-Fail $suite "compile math functions" }

        $errorResp = Invoke-RestMethod -Method Post -Uri ("http://localhost:{0}/compile" -f $port) -ContentType "application/json" -Body '{"code":"(define incomplete","session_id":"test4"}'
        if ($errorResp.success -eq $false) { Add-Pass $suite } else { Add-Fail $suite "error handling" }

        $indexResp = Invoke-WebRequest -Uri ("http://localhost:{0}/" -f $port) -UseBasicParsing
        if ($indexResp.Content -match "Eshkol REPL") { Add-Pass $suite } else { Add-Fail $suite "static index.html" }

        $styleResp = Invoke-WebRequest -Uri ("http://localhost:{0}/style.css" -f $port) -UseBasicParsing
        if ($styleResp.Content -match "Eshkol REPL Styles") { Add-Pass $suite } else { Add-Fail $suite "static style.css" }

        $jsResp = Invoke-WebRequest -Uri ("http://localhost:{0}/eshkol-repl.js" -f $port) -UseBasicParsing
        if ($jsResp.Content -match "class EshkolRepl") { Add-Pass $suite } else { Add-Fail $suite "static eshkol-repl.js" }

        $wasmResp = Invoke-RestMethod -Method Post -Uri ("http://localhost:{0}/compile" -f $port) -ContentType "application/json" -Body '{"code":"(define x 42)","session_id":"test5"}'
        if ($wasmResp.wasm -match "^AGFzbQ") { Add-Pass $suite } else { Add-Fail $suite "WASM binary response" }

        $multiResp = Invoke-RestMethod -Method Post -Uri ("http://localhost:{0}/compile" -f $port) -ContentType "application/json" -Body '{"code":"(define a 1)\n(define b 2)\n(define (add x y) (+ x y))","session_id":"test6"}'
        if ($multiResp.success -eq $true) { Add-Pass $suite } else { Add-Fail $suite "multiple definitions" }
    } finally {
        if ($server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
            $server.WaitForExit()
        }
        $server.Dispose()
    }

    Show-SuiteSummary $suite
    return $suite
}

function Invoke-XlaSuite {
    Write-Section "Eshkol XLA/StableHLO Integration Tests"
    $suite = New-SuiteState "xla"

    $xlaBin = Join-Path $script:BinaryDir "xla_codegen_test.exe"
    if (Test-Path $xlaBin) {
        $run = Invoke-ProcessCapture -FilePath $xlaBin -WorkingDirectory $script:ProjectRoot
        if ($run.ExitCode -eq 0) {
            Add-Pass $suite
        } else {
            Add-Fail $suite "C++ Unit Tests"
        }
    } else {
        Add-Skip $suite "C++ Unit Tests skipped (xla_codegen_test.exe not built)."
    }

    $files = Get-TestFiles -ProjectRoot $script:ProjectRoot -Patterns @("tests/xla/*.esk")
    foreach ($testFile in $files) {
        $testName = Split-Path -Leaf $testFile
        $outputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "xla" -TestName $testName
        $compile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $testFile -OutputBase $outputBase
        if (-not $compile.Success) {
            Format-TestStatus $testName "COMPILE FAIL" Red
            Add-Fail $suite $testName
            continue
        }
        $run = Invoke-ProcessCapture -FilePath $compile.ExePath -WorkingDirectory $script:ProjectRoot -TimeoutSec 60
        if ($run.TimedOut) {
            Format-TestStatus $testName "TIMEOUT" Red
            Add-Fail $suite "$testName (timeout)"
        } elseif ($run.ExitCode -ne 0) {
            Format-TestStatus $testName ("RUNTIME FAIL (exit {0})" -f $run.ExitCode) Red
            Add-Fail $suite $testName
        } elseif ($run.Output -match "^FAIL:" -or ($run.Output -notmatch "PASS" -and $run.Output -match "(?i)error:")) {
            Format-TestStatus $testName "ASSERTION FAIL" Red
            Add-Fail $suite $testName
        } else {
            Format-TestStatus $testName "PASS" Green
            Add-Pass $suite
        }
    }

    $perfSource = Join-Path $script:TempRoot "xla_perf_test.esk"
    @'
;; Quick performance sanity check
(define N 150)
(define A (reshape (arange (* N N)) N N))
(define B (reshape (ones (* N N)) N N))
(define C (matmul A B))
(display "150x150 matmul: ")
(display (tensor-shape C))
(newline)
'@ | Set-Content -Path $perfSource -Encoding ASCII

    $perfOutputBase = New-OutputBase -TempRoot $script:TempRoot -SuiteName "xla" -TestName "performance_sanity"
    $perfCompile = Invoke-EshkolCompile -EshkolRun $script:EshkolRun -ProjectRoot $script:ProjectRoot -BuildDir $script:BuildDir -TestFile $perfSource -OutputBase $perfOutputBase
    if (-not $perfCompile.Success) {
        Add-Fail $suite "performance_sanity"
    } else {
        $perfRun = Invoke-ProcessCapture -FilePath $perfCompile.ExePath -WorkingDirectory $script:ProjectRoot -TimeoutSec 60
        if ($perfRun.TimedOut -or $perfRun.ExitCode -ne 0) {
            Add-Fail $suite "performance_sanity"
        } else {
            Add-Pass $suite
        }
    }

    Show-SuiteSummary $suite "XLA Test Results Summary"
    return $suite
}

function Ensure-BuildArtifacts {
    param([string[]]$Targets)
    $missingTargets = @()
    foreach ($target in $Targets) {
        $path = Join-Path $script:BinaryDir ($target + ".exe")
        if ($target -eq "stdlib") {
            $path = Join-Path $script:BuildDir "stdlib.o"
            if (-not (Test-Path $path)) {
                $path = Join-Path $script:BinaryDir "stdlib.o"
            }
        }
        if (-not (Test-Path $path)) {
            $missingTargets += $target
        }
    }

    if ($missingTargets.Count -eq 0) {
        return
    }

    Write-Host ("Building missing targets: {0}" -f ($missingTargets -join ", ")) -ForegroundColor Yellow
    & cmake --build $script:BuildDir --target @($missingTargets) --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build required targets: $($missingTargets -join ', ')"
    }
}

$script:ProjectRoot = Get-ProjectRoot
$buildLayout = Resolve-BuildDirectory -ProjectRoot $script:ProjectRoot -RequestedBuildDir $BuildDir
$script:BuildDir = $buildLayout.RootDir
$script:BinaryDir = $buildLayout.BinaryDir
$script:TempRoot = Join-Path $env:TEMP "eshkol-windows-suite"
Ensure-NativeTempRoot -TempRoot $script:TempRoot

$script:EshkolRun = Join-Path $script:BinaryDir "eshkol-run.exe"
$script:EshkolRepl = Join-Path $script:BinaryDir "eshkol-repl.exe"
$script:EshkolServer = Join-Path $script:BinaryDir "eshkol-server.exe"
$env:ESHKOL_PATH = Join-Path $script:ProjectRoot "lib"

if (-not $SkipConfigureBuild) {
    Ensure-BuildArtifacts -Targets @("eshkol-run", "eshkol-repl", "stdlib")
}

Write-Section "Eshkol Complete Windows Test Suite"
Write-Host ("Project Root: {0}" -f $script:ProjectRoot)
Write-Host ("Build Dir:    {0}" -f $script:BuildDir)
Write-Host ("Binary Dir:   {0}" -f $script:BinaryDir)
Write-Host ("Mode:         {0}" -f $Mode)
Write-Host ""

$suiteResults = @()
switch ($Mode) {
    "all" {
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "features" -Title "Eshkol Features Test Suite" -Patterns @("tests/features/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "stdlib" -Title "Eshkol Stdlib Test Suite" -Patterns @("tests/stdlib/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "list" -Title "Eshkol List Test Suite" -Patterns @("tests/list/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-MemorySuite
        $suiteResults += Invoke-ModulesSuite
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "types" -Title "Eshkol HoTT Type System Test Suite" -Patterns @("tests/types/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-TypesystemSuite
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "autodiff" -Title "Eshkol Autodiff Test Suite" -Patterns @("tests/autodiff/*.esk", "tests/autodiff_debug/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "ml" -Title "Eshkol ML Test Suite" -Patterns @("tests/ml/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "neural" -Title "Eshkol Neural Network Test Suite" -Patterns @("tests/neural/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "json" -Title "JSON Test Suite Validation" -Patterns @("tests/json/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "system" -Title "System Test Suite" -Patterns @("tests/system/*.esk")
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "complex" -Title "Eshkol Complex & FFT Test Suite" -Patterns @("tests/complex/*.esk") -RuntimeErrorRegex "error:"
        $suiteResults += Invoke-CppTypeSuite
        $suiteResults += Invoke-ParserSuite
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "control_flow" -Title "Eshkol Control Flow Test Suite" -Patterns @("tests/control_flow/*.esk") -FailRegex "FAIL:"
        $suiteResults += Invoke-LogicSuite
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "bignum" -Title "Eshkol Bignum Tests" -Patterns @("tests/bignum/*.esk") -FailRegex "^FAIL:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "rational" -Title "Eshkol Rational Number Tests" -Patterns @("tests/rational/*.esk") -FailRegex "^FAIL:"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "parallel" -Title "Eshkol Parallel Primitives Tests" -Patterns @("tests/parallel/*.esk") -FailRegex "^FAIL:|Failed:\s+[1-9]"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "signal" -Title "Eshkol Signal Processing Tests" -Patterns @("tests/signal/*.esk") -FailRegex "^FAIL:"
        $suiteResults += Invoke-TimedCompileRunSuite -SuiteName "optimization" -Title "Eshkol Optimization Algorithms Tests" -Patterns @("tests/ml/*optimization*.esk") -TimeoutSec 60 -FailRegex "FAIL:"
        $suiteResults += Invoke-ExamplesSuite
        $suiteResults += Invoke-XlaSuite
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "gpu" -Title "Eshkol GPU Test Suite" -Patterns @("tests/gpu/*.esk") -FailRegex "^FAIL:|Failed:\s+[1-9]"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "error_handling" -Title "Eshkol Error Handling Tests" -Patterns @("tests/error_handling/*.esk") -FailRegex "^FAIL:|Failed:\s+[1-9]"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "macros" -Title "Eshkol Macros Tests" -Patterns @("tests/macros/*.esk") -FailRegex "^FAIL:|Failed:\s+[1-9]"
        $suiteResults += Invoke-ReplSuite
        $suiteResults += Invoke-WebSuite
        $suiteResults += Invoke-TimedCompileRunSuite -SuiteName "tco" -Title "Eshkol TCO (Tail Call Optimization) Tests" -Patterns @("tests/tco/*.esk") -TimeoutSec 60 -FailRegex "FAIL" -RequirePassMarker
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "io" -Title "Eshkol I/O Test Suite" -Patterns @("tests/io/*.esk") -FailRegex "^FAIL" -TimeoutSec 10
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "benchmark" -Title "Eshkol Benchmark Test Suite" -Patterns @("tests/benchmark/*.esk", "tests/benchmarks/*.esk") -FailRegex "^FAIL"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "migration" -Title "Eshkol Migration Test Suite" -Patterns @("tests/migration/*.esk") -FailRegex "^FAIL"
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "codegen" -Title "Eshkol Codegen Test Suite" -Patterns @("tests/codegen/*.esk") -FailRegex "^FAIL" -Recurse
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "numeric" -Title "Eshkol Numeric Regression Tests" -Patterns @("tests/numeric/*.esk") -FailRegex "^FAIL:"
    }
    "xla" {
        $suiteResults += Invoke-XlaSuite
    }
    "gpu" {
        $suiteResults += Invoke-SimpleCompileRunSuite -SuiteName "gpu" -Title "Eshkol GPU Test Suite" -Patterns @("tests/gpu/*.esk") -FailRegex "^FAIL:|Failed:\s+[1-9]"
    }
}

$suiteResults = @(
    $suiteResults |
    Where-Object { $_ -and $_.PSObject.Properties.Match("Failed").Count -gt 0 }
)

$totalSuites = $suiteResults.Count
$failedSuites = @($suiteResults | Where-Object { $_.Failed -gt 0 })
$passedSuites = @($suiteResults | Where-Object { $_.Failed -eq 0 })
$totalPassed = ($suiteResults | Measure-Object -Property Passed -Sum).Sum
$totalFailed = ($suiteResults | Measure-Object -Property Failed -Sum).Sum
$totalSkipped = ($suiteResults | Measure-Object -Property Skipped -Sum).Sum

Write-Section "Complete Test Suite Summary"
Write-Host ("Total Suites Run:   {0}" -f $totalSuites)
Write-Host ("Suites Passed:      {0}" -f $passedSuites.Count) -ForegroundColor Green
Write-Host ("Suites Failed:      {0}" -f $failedSuites.Count) -ForegroundColor Red
Write-Host ("Tests Passed:       {0}" -f $totalPassed) -ForegroundColor Green
Write-Host ("Tests Failed:       {0}" -f $totalFailed) -ForegroundColor Red
Write-Host ("Tests Skipped:      {0}" -f $totalSkipped) -ForegroundColor Yellow
Write-Host ""

if ($failedSuites.Count -gt 0) {
    Write-Host "Failed Suites:" -ForegroundColor Red
    foreach ($suite in $failedSuites) {
        Write-Host ("  - {0}" -f $suite.Name)
    }
    Write-Host ""
    exit 1
}

Write-Host "All Windows suites passed." -ForegroundColor Green
exit 0
