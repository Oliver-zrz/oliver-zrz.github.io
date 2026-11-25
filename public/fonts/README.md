# 网站字体

Noto Sans SC 来自 Google Fonts，使用 SIL Open Font License 1.1。许可证保存在 `OFL-NotoSansSC.txt`，来源和文件校验和见 `font-info.json`。

本站通过同一域名提供 WOFF2 字体，英文和中文分开加载，字重范围 400–600。采用 `font-display: swap`，下载期间先显示系统字体。正文、标题和导航都使用无衬线字体；代码单独使用等宽字体。

中文子集包含 GB2312 和制作时的站点文字，未包含的字使用系统字体回退。后续文章出现生僻字时可按需扩充，不要求在每次网站构建时运行字体工具或访问字体服务。

字体是已经准备好的静态资源；正常执行 `pnpm build` 不需要安装 Python、fontTools 或 Brotli。
