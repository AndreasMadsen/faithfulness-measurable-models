
.PHONY: upload-code-graham

sync: upload-code-graham

upload-code-graham:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-graham:~/workspace/economical-roar

upload-code-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' \
		-e ssh ./ cc-narval:~/workspace/economical-roar
