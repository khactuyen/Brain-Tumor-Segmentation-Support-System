$word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
try {
    foreach ($doc in $word.Documents) {
        [pscustomobject]@{
            Name = $doc.Name
            FullName = $doc.FullName
            Saved = $doc.Saved
            ReadOnly = $doc.ReadOnly
        } | Format-List
    }
}
finally {
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
}
