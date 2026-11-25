const modules = import.meta.glob('../pages/blog/*.md', { eager: true });
export const posts = Object.values(modules).map((entry: any) => ({
  ...entry.frontmatter,
  url: entry.url,
})).sort((a, b) => String(b.date).localeCompare(String(a.date)));
export const dateLabel = (value: string) => String(value).slice(0, 10).replaceAll('-', '.');
