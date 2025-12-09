# How to Make These Docs

This documentation is built using [MkDocs](https://www.mkdocs.org/), a static site generator designed for project documentation. The documentation source files are written in Markdown and stored in the `docs/` directory.

## What is MkDocs?

MkDocs is a fast, simple static site generator that's geared toward building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file (`mkdocs.yml`).

## Documentation Structure

```
napari-easy-augment-batch-dl/
├── mkdocs.yml           # Configuration file
├── docs/                # Documentation source files
│   ├── index.md         # Home page
│   ├── overview.md
│   ├── load.md
│   ├── faq.md
│   └── ...
└── site/                # Generated site (git-ignored)
```

## Installing MkDocs

MkDocs should be installed in your Python environment:

```bash
pip install mkdocs
pip install mkdocs-material  # Material theme used in this project
```

## Key Commands

### `mkdocs serve`

Preview the documentation locally with live reloading:

```bash
mkdocs serve
```

This starts a local development server at `http://127.0.0.1:8000/`. The site will automatically rebuild and refresh when you save changes to any Markdown files.

**Usage in this repo:**
1. Navigate to the repository root directory
2. Run `mkdocs serve`
3. Open your browser to the displayed URL
4. Edit any `.md` files in the `docs/` directory
5. See changes reflected immediately in your browser

### `mkdocs build`

Build the static site to the `site/` directory:

```bash
mkdocs build
```

This generates the static HTML/CSS/JavaScript files that can be deployed to any web server. The `site/` directory is typically git-ignored.

**Usage in this repo:**
- Usually not needed manually, as `mkdocs gh-deploy` handles building
- Useful for testing the final build output
- Can verify all links and references are correct

### `mkdocs gh-deploy`

Build and deploy the documentation to GitHub Pages:

```bash
mkdocs gh-deploy
```

This command:
1. Builds the documentation with `mkdocs build`
2. Commits the built site to the `gh-pages` branch
3. Pushes the branch to GitHub
4. Your docs are automatically published to GitHub Pages

**Usage in this repo:**
1. Make sure all changes are committed to your main branch
2. Run `mkdocs gh-deploy` from the repository root
3. GitHub Pages will serve the docs at: `https://True-North-Intelligent-Algorithms.github.io/napari-easy-augment-batch-dl/`

!!! note
    You need write access to the repository to deploy to GitHub Pages.

## Configuration File: `mkdocs.yml`

The `mkdocs.yml` file at the repository root controls:

- **Site name and theme** (Material theme is used)
- **Navigation structure** (order of pages in sidebar)
- **Markdown extensions** (for admonitions, code blocks, emoji)
- **Repository URL** (for "Edit on GitHub" links)

### Adding a New Page

1. Create a new `.md` file in the `docs/` directory:
   ```
   docs/my_new_page.md
   ```

2. Add it to the navigation in `mkdocs.yml`:
   ```yaml
   nav:
     - Home: index.md
     - Overview: overview.md
     - My New Page: my_new_page.md  # Add here
     - FAQ: faq.md
   ```

3. The page will appear in the sidebar in the order specified

## Markdown Features

This documentation uses several Markdown extensions:

### Admonitions (Note/Warning Boxes)

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.
```

### Code Blocks

```markdown
```python
def example():
    print("Hello, world!")
```
```

### Images

```markdown
![Alt text](images/screenshot.png)
```

Images should be placed in `docs/images/` directory.

### Links

- Internal links: `[Link text](other_page.md)`
- External links: `[Link text](https://example.com)`
- Anchor links: `[Link to section](#section-heading)`

## Workflow for Updating Documentation

1. **Edit locally:**
   ```bash
   mkdocs serve
   # Edit .md files in docs/
   # Preview changes in browser
   ```

2. **Commit changes:**
   ```bash
   git add docs/
   git commit -m "Update documentation"
   git push origin main
   ```

3. **Deploy to GitHub Pages:**
   ```bash
   mkdocs gh-deploy
   ```

## Tips

- Use `mkdocs serve` frequently while writing to see changes in real-time
- Keep documentation files focused and concise
- Use screenshots to illustrate UI elements (store in `docs/images/`)
- Use admonitions to highlight important information
- Test all internal links before deploying
- The Material theme provides search functionality automatically

## Troubleshooting

**Problem:** `mkdocs: command not found`
- **Solution:** Install MkDocs: `pip install mkdocs mkdocs-material`

**Problem:** Changes not appearing when using `mkdocs serve`
- **Solution:** Check the terminal for errors, ensure you're editing files in `docs/` not `site/`

**Problem:** `mkdocs gh-deploy` fails
- **Solution:** Ensure you have push access to the repository and the `gh-pages` branch exists

**Problem:** Images not displaying
- **Solution:** Use relative paths from the Markdown file: `images/screenshot.png`

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)
