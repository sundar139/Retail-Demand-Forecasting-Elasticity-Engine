param(
    [string]$InputPath,
    [double]$TrainRatio = 0.70,
    [double]$ValidationRatio = 0.15,
    [double]$TestRatio = 0.15,
    [string]$ValidationStart,
    [string]$TestStart
)

$command = @(
    "run",
    "--python",
    "3.13",
    "retail-forecasting-engine",
    "prepare-data",
    "--train-ratio",
    $TrainRatio,
    "--validation-ratio",
    $ValidationRatio,
    "--test-ratio",
    $TestRatio
)

if ($InputPath) {
    $command += @("--input-path", $InputPath)
}
if ($ValidationStart) {
    $command += @("--validation-start", $ValidationStart)
}
if ($TestStart) {
    $command += @("--test-start", $TestStart)
}

uv @command
