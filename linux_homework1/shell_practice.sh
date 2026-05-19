#!/bin/bash
mkdir -p linux_practice/{docs,backup}
touch linux_practice/docs/readme.txt linux_practice/docs/notes.log linux_practice/docs/temp.tmp
rm linux_practice/docs/temp.tmp
mv linux_practice/docs/notes.log linux_practice/docs/daily_report.txt
echo "Project Status: Active" > linux_practice/docs/daily_report.txt
date >> linux_practice/docs/daily_report.txt
cp linux_practice/docs/*.txt linux_practice/backup/
chmod 444 linux_practice/backup/*
for file in linux_practice/backup/*; do
    if [ -f "$file" ]; then
        echo "Archive Complete. File [$file] is now read-only"
    fi
done
