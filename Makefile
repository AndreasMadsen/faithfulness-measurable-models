
.PHONY: noop download-code-narval upload-code-narval upload-code-cedar upload-code-mila

noop:
	echo "preventing default action"

download-results-narval:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-narval:~/scratch/ecoroar/results/ ./results

upload-code-narval:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-narval:~/workspace/economical-roar

upload-code-cedar:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-cedar:~/workspace/economical-roar

upload-code-mila:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ mila:~/workspace/economical-roar
