# Runze Zhang · 开发手记

个人博客：[oliver-zrz.github.io](https://oliver-zrz.github.io)。Astro 静态网站，通过 GitHub Actions 发布到 GitHub Pages。

## 目录

```text
src/pages/blog/       # 从私有 dev-home 导出的公开文章
src/layouts/         # 页面与文章布局
src/data/skills.json # 公开的 Skill 名称与用途目录
src/styles/          # 响应式样式和明暗主题
public/media/        # 每篇文章列明的公开配图
content-sync.json    # 导出器管理的文件及其校验和
.github/workflows/   # GitHub Pages 构建与部署
```

## 开发

需要 Node.js 22.23.4+ 和 pnpm。

```bash
pnpm install --frozen-lockfile
pnpm dev
pnpm build
```

文章正文的唯一来源在私有 dev-home。修改文章后从那里运行 `python tools/devhome.py export --site <本仓库路径>`，再检查 diff、构建并提交。公开仓库不保存完整的私有工作日志、登录凭据和环境配置。
