# Variables
DOCS_DIR = docs

# Targets
.PHONY: docs clean

# Build the JupyterBook documentation
docs:
	jupyter-book build $(DOCS_DIR)

# Clean the generated build files
clean:
	rm -rf $(DOCS_DIR)/_build

# Preview the built documentation locally
serve:
	cd $(DOCS_DIR)/_build/html && python -m http.server
