param(
    [string]$File = "01_HS.out"
)

Write-Host "========================================"
Write-Host " ORCA OUTPUT CHECK: $File"
Write-Host "========================================"

Write-Host "`n[1] TERMINATION"
Select-String "ORCA TERMINATED NORMALLY" $File

Write-Host "`n[2] FINAL ENERGY"
Select-String "FINAL SINGLE POINT ENERGY" $File | Select-Object -Last 1

Write-Host "`n[3] <S^2>"
Select-String "Expectation value" $File | Select-Object -Last 3

Write-Host "`n[4] TOTAL ATOMIC SPIN"
Select-String "Sum of atomic spin populations" $File | Select-Object -Last 1

Write-Host "`n[5] MULLIKEN SPIN POPULATION"
Select-String "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS" $File -Context 0,15

Write-Host "`n[6] HOMO / LUMO / GAP"
Select-String "HOMO|LUMO|GAP" $File

Write-Host "`n========================================"