# Auto-detect torch version and print (or install) matching PyG wheels
param(
    [switch]$Install
)

$py = "python"
try {
    $info = & $py -c "import torch,sys; print(torch.__version__); print(torch.version.cuda if hasattr(torch.version, 'cuda') else 'None')" 2>$null
} catch {
    Write-Error "Could not invoke python to detect torch. Ensure python is in PATH or pass full path to python."; exit 1
}

if (-not $info) {
    Write-Error "Failed to detect torch version."; exit 1
}

$lines = $info -split "\n"
$torch_ver = $lines[0].Trim()
$cuda_ver = if ($lines.Length -gt 1) { $lines[1].Trim() } else { 'None' }

if ($cuda_ver -eq 'None' -or $cuda_ver -eq '' -or $cuda_ver -eq 'NoneType') { $tag = 'cpu' } else { $tag = 'cu' + ($cuda_ver -replace '\.', '') }

Write-Host "Detected torch==$torch_ver, cuda="$cuda_ver
$wheel_index = "https://data.pyg.org/whl/torch-$torch_ver+$tag.html"
Write-Host "Using PyG wheel index: $wheel_index"

$packages = @('torch-scatter','torch-sparse','torch-cluster','torch-spline-conv','torch-geometric')

$cmds = @()
foreach ($p in $packages) {
    $cmds += "python -m pip install --no-cache-dir -U \"$p\" -f $wheel_index"
}

Write-Host "Suggested pip commands to install PyG components (run with -Install to execute):"
foreach ($c in $cmds) { Write-Host $c }

if ($Install) {
    foreach ($c in $cmds) {
        Write-Host "Running: $c"
        iex $c
+    }
 }
