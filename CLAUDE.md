# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a personal academic portfolio website for Ashod Khederlarian (PhD student, University of Pittsburgh), built on the [Prologue HTML5 UP template](https://html5up.net/prologue) and powered by Jekyll. It is deployed via GitHub Pages at `ashodkh.github.io`.

## Editing the Site

Key files:
- `index.html` — home page (intro, about, contact sections)
- `research.html` — research listing page (brief cards with "Read more" links)
- `research/emission-lines.html` — full emission lines project page
- `research/photo-zs.html` — full photo-z project page
- `blog.html` — blog listing page (brief cards with "Read more" links)
- `blog/gravitational-waves.html` — full gravitational waves blog post
- `_layouts/default.html` — shared layout with top nav
- `portfolio-2.html` — untracked, work in progress

Images live in `images/`. Reference them with `{{ '/images/filename.ext' | relative_url }}`.

## CSS / Styles

- **Do not edit `assets/css/main.css` directly** — it is compiled from SCSS.
- Source styles are in `assets/sass/main.scss` (imports from `assets/sass/libs/`).
- To recompile CSS after editing SCSS:
  ```
  sass assets/sass/main.scss assets/css/main.css
  ```

## Page Structure

### `index.html`

| Section ID | Content       |
|------------|---------------|
| `#intro`   | Intro / hero  |
| `#about`   | About Me      |
| `#contact` | Contact       |

### `research.html` (listing page)

| Section / ID     | Content                                                        |
|------------------|----------------------------------------------------------------|
| `#emission_lines`| Brief card — JAX neural network / emission lines project       |
| `#photo_zs`      | Brief card — Semi-supervised photo-z deep learning project     |

### `blog.html` (listing page)

| Section / ID | Content                                               |
|--------------|-------------------------------------------------------|
| `#GWs`       | Brief card — Gravitational wave detection blog post   |

### Detail pages

Each listing card links to a full-content page:
- `research/emission-lines.html` — full emission lines content
- `research/photo-zs.html` — full photo-z content
- `blog/gravitational-waves.html` — full GW blog post

### Navigation (`_layouts/default.html`)

The top nav links are:
- **Research** → `/research`
- **About Me** → `/#about`
- **Contact** → `/#contact`

To add a new research project: add a `<div class="project" id="new_id">` block in `research.html`.

## Previewing Locally

The site uses Jekyll. To serve locally:
```
bundle exec jekyll serve
```
Then visit `http://localhost:4000`.

First-time setup (requires rbenv with Ruby 3.3+):
```
rbenv local 3.3.0
gem install bundler
bundle install
```

## Deployment

Push to the `master` branch; GitHub Pages serves the site automatically from the repo root.
