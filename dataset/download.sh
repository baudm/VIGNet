#!/usr/bin/env sh

# https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805#gistcomment-2316906
download() {
	FILEID="$1"
	FILENAME="$2"
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O "$FILENAME" && rm -rf /tmp/cookies.txt
}

download '1I3pjmSSyhJRDW6zYchb1KOfqdmM0llWd' motorcycle.train.npz
download '1x-Gj8f_sJjkEJ9gN1VpI3x161paf4m8i' motorcycle.test.npz
download '1RJGPjFUvYwDXxW40Ml8LsjJchV04N4fP' car.train.npz
download '1gfVBwoBVnbCnjZpPt8cxIaC7jpXoBZsR' car.test.npz
