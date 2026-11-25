# Personal blog

Keep this repository public-safe. Blog Markdown, article media and the Skill catalog are generated from the selected public content in dev-home; edit the canonical private source first, then export. Do not copy whole private folders here.

The site is static Astro with no server, login, analytics, or external font dependency. Preserve readable Chinese text, accessible links and keyboard controls. Run `pnpm build` after source changes. Verify article images, responsive layout, search filters, theme switching, RSS and links before deployment.

The owner prefers compact pages, cohesive color, and sans-serif typography. Songti/serif headings, multicolor cards/tags, and an almost monochrome replacement were rejected. Headings, body text and secondary information must have clear color differences, with a small number of coordinated hues reused by role. The approved design uses blue-violet for headings/links, rose for section/category labels, and neutral body text and surfaces. Use Josh W. Comeau's homepage and interop-2027 article as references for this hierarchy. Keep introductions short and articles visible in the first desktop viewport. Do not shrink body text to make the layout denser. Respect the requested preview/publication scope; this design was reviewed locally and then explicitly approved for publication.

Typography uses locally hosted, OFL-licensed Noto Sans SC subsets, with a 400–600 variable weight range and system sans-serif fallbacks. Keep code monospaced. Font files, coverage information and license are in `public/fonts/`; no third-party font requests are needed at runtime.

Visitor-facing toolbox names and summaries live in `src/data/toolbox.ts`. Keep the original Skill identifiers accurate; do not show agent trigger instructions as website descriptions. Write direct, factual Chinese prose and avoid repeated manifestos, generic conclusions or descriptions of the backstage writing workflow.
