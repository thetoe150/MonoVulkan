Get-ChildItem -Path ./build/ -Include *.* -File -Recurse | foreach { $_.Delete()}
