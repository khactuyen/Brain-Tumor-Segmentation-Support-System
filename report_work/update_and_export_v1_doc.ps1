param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf
)

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($InputDocx, $false, $false)

    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    foreach ($story in $doc.StoryRanges) {
        $range = $story
        while ($null -ne $range) {
            $range.Fields.Update() | Out-Null
            $range = $range.NextStoryRange
        }
    }

    $doc.Repaginate()
    $doc.Save()
    $doc.ExportAsFixedFormat($OutputPdf, 17)
    Write-Output "pages=$($doc.ComputeStatistics(2))"
}
finally {
    if ($null -ne $doc) {
        $doc.Close(0)
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) | Out-Null
    }
    if ($null -ne $word) {
        $word.Quit()
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
