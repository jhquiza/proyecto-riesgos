---
name: feedback_tectonic
description: User compiles LaTeX with tectonic, not pdflatex/MacTeX
type: feedback
---

Use `tectonic tesis.tex` to compile the thesis PDF, not pdflatex.
**Why:** User has tectonic installed at /opt/homebrew/bin/tectonic, no other LaTeX distribution.
**How to apply:** When compiling LaTeX documents, use `tectonic` directly. Do not attempt to install MacTeX/BasicTeX.
