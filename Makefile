# Book metadata
TITLE="Training ImageNets with Orange"
FILEBASE=book
SITE=public
include chapters.conf

# HTML only: the finished site is public/ (public/index.html is the site root).
all: html
html: $(SITE)/index.html

$(FILEBASE).md: $(INPUT) utils/fix_links.py
	rm -f $(FILEBASE).md
	for file in $(INPUT); do \
		cat "$$file" >> $(FILEBASE).md; \
		printf '\n<div style="page-break-before: always;"></div>\n\n' >> $(FILEBASE).md; \
	done
	utils/fix_links.py < $(FILEBASE).md > $(FILEBASE).md2
	mv $(FILEBASE).md2 $(FILEBASE).md

$(SITE)/index.html: $(FILEBASE).md css/site.css $(wildcard imgs/*)
	mkdir -p $(SITE)/imgs $(SITE)/css
	cp css/* $(SITE)/css/
	cp imgs/* $(SITE)/imgs/ 2>/dev/null || true
	find chapters/ \( -name '*.png' -o -name '*.jpg' -o -name '*.gif' -o -name '*.jpeg' \) -exec cp "{}" $(SITE)/imgs/ \;
	pandoc $(FILEBASE).md -o $(SITE)/index.html \
		--metadata title=$(TITLE) \
		--css css/site.css \
		--toc --toc-depth=2 \
		--standalone

clean:
	rm -rf $(SITE) $(FILEBASE).md

# Build the image datasets (needs the Orange env with keras + torchvision; EMNIST is a 562 MB download).
datasets:
	python utils/export_datasets.py

.PHONY: all html clean datasets
