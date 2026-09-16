param([Parameter(Mandatory = $true)][string]$TargetPath)

$resolvedTarget = [IO.Path]::GetFullPath($TargetPath)
$word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
$closed = $false
try {
    foreach ($doc in @($word.Documents)) {
        if ([string]::Equals([IO.Path]::GetFullPath($doc.FullName), $resolvedTarget, [StringComparison]::OrdinalIgnoreCase)) {
            if (-not $doc.Saved) {
                throw "Target document has unsaved changes; refusing to close it."
            }
            $doc.Close(0)
            $closed = $true
            break
        }
    }
    if (-not $closed) {
        throw "Target document was not found in the running Word instance."
    }
    if ($word.Documents.Count -eq 0) {
        $word.Quit()
    }
}
finally {
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
}

Write-Output "Closed saved target document: $resolvedTarget"
