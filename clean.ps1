Get-ChildItem -Path ./obj/ -Include *.* -File -Recurse | foreach { $_.Delete()}
