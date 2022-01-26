
.PHONY: sync upload-code-narval download-code-narval upload-code-graham download-code-graham

noop:
	echo "preventing default action"

upload-code-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-narval:~/workspace/economical-roar

download-code-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh cc-narval:~/workspace/economical-roar/ .

upload-code-graham:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-graham:~/workspace/economical-roar

download-code-graham:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh cc-graham:~/workspace/economical-roar/ .
